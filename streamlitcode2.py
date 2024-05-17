import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import correlate
import os
import platform


def capture_consecutive_frames():
    cap = cv2.VideoCapture(0) 
    #cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    gray_frames = []
    for _ in range(2):  # Capture 2 consecutive frames
        ret, frame = cap.read()
        if ret:
            gray_frames.append(frame)
        else:
            print("Error: Unable to capture frame.")
    cap.release()
    return gray_frames

def vel_field(curr_frame, next_frame, win_size):
    ys = np.arange(0, curr_frame.shape[0], win_size)
    xs = np.arange(0, curr_frame.shape[1], win_size)
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            int_win = curr_frame[y : y + win_size, x : x + win_size]
            search_win = next_frame[y : y + win_size, x : x + win_size]
            cross_corr = correlate(
                search_win - search_win.mean(), int_win - int_win.mean(), method="fft"
            )
            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                - np.array([win_size, win_size])
                + 1
            )
    ys = ys + win_size / 2
    xs = xs + win_size / 2
    return xs, ys, dxs, dys

def background_subtraction_live():
    cap = cv2.VideoCapture(0)
    fp = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in video =', fp)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'Height: {height}, Width: {width}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps:0.2f}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background = first_gray
    count = 1
    meanframe = cv2.absdiff(first_gray, background)
    while ret:
        first = cv2.absdiff(first_gray, background)
        ret, first = cv2.threshold(first, 15, 255, cv2.THRESH_TOZERO)
        ret, first = cv2.threshold(first, 80, 255, cv2.THRESH_TOZERO_INV)
        first = first * 4
        meanframe = cv2.addWeighted(meanframe, 1 - 1 / count, first, 1 / count, 0)
        count += 1
        first = cv2.absdiff(first, meanframe)
        first = cv2.blur(first, (5, 5))
        first = cv2.GaussianBlur(first, (5, 5), 0)
        first = cv2.medianBlur(first, 5)
        #frame1 = cv2.applyColorMap(first, cv2.COLORMAP_JET)
        cv2.imshow("difference", first)
        background = first_gray
        ret, first_frame = cap.read()
        first_gray=cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            break 
    cap.release()
    cv2.destroyAllWindows() 
    
    
    
def background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    # Get video properties
    fp = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total frames in video =', fp)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'Height: {height}, Width: {width}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps:0.2f}')

    # Set initial frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        exit()

    # Initialize variables for background subtraction
    background = img
    count = 1
    meanframe = cv2.absdiff(img, background)

    # Initialize video writer
    output_video_path = os.path.join(os.getcwd(), 'stable_Background.avi')
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))

    while ret:
        frame = cv2.absdiff(img, background)
        ret, frame = cv2.threshold(frame, 15, 255, cv2.THRESH_TOZERO)
        ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_TOZERO_INV)
        frame = frame * 4
        meanframe = cv2.addWeighted(meanframe, 1 - 1 / count, frame, 1 / count, 0)
        count += 1
        frame = cv2.absdiff(frame, meanframe)
        frame = cv2.blur(frame, (5, 5))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.medianBlur(frame, 5)
        frame1 = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        writer.write(frame1)
        background=img
        ret, img = cap.read()
        if not ret:
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for the output video file
    output_video_path = os.path.join(os.getcwd(), 'output_opticalflow_apoorv_v6.avi')
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height))

    # Read the first frame of the video
    ret, frame1 = cap.read()

    # Convert the first frame to grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Create an HSV image with all values set to 0
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set image saturation to maximum

    # Loop through the rest of the frames in the video
    while True:
        # Read the next frame of the video
        ret, frame2 = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Convert the current frame to grayscale
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow between the two frames using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 7, 1.2, 0)

        # Calculate the magnitude and angle of each 2D vector in flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Convert angle to hue value and magnitude to value value for each pixel in hsv image
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert hsv image to bgr image
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr1 = cv2.applyColorMap(bgr * 5, cv2.COLORMAP_PLASMA)

        # Write the current frame to output video file
        writer.write(bgr1)

        # Set prvs to next for next iteration of loop
        prvs = next

    # Release VideoCapture and VideoWriter objects and close all windows
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def main():
    st.image("header_image.jpg", use_column_width=True)
    st.title("WELCOME TO WEBSchlieren (BOS)")
    st.title("visualize the invisible ")

    if st.checkbox("live velocity field visualization"):
        selected_techniques = st.multiselect("Select image processing techniques:",
                                     ["Cross Correlation", "background subtraction"])

        if "Cross Correlation" in selected_techniques:
            st.write("Capturing frames...")
            start_recording = st.checkbox("Start Recording")
            if start_recording:
                gray_frames = capture_consecutive_frames()
                a = cv2.cvtColor(gray_frames[0], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(gray_frames[1], cv2.COLOR_BGR2GRAY)

                if len(gray_frames) == 2:
                    st.write("Frames captured successfully!")
                    st.image(a, caption="First Frame", channels="GRAY")
                    st.image(b, caption="Second Frame", channels="GRAY")
                    
                    resized_a = cv2.resize(a, (256, 512))
                    resized_b = cv2.resize(b, (256, 512))
                    xs, ys, dxs, dys = vel_field(resized_a, resized_b, 16)
                    norm_drs = np.sqrt(dxs ** 2 + dys ** 2)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.quiver(
                            xs,
                            ys[::-1],
                            dxs,
                            -dys,
                            norm_drs,
                            cmap="plasma",
                            angles="xy",
                            scale_units="xy",
                            scale=0.25,
                        )
                    ax.set_aspect("equal")
                    st.pyplot(fig)
                    
                else:
                    st.write("Error: Unable to capture frames.")
        if "background subtraction" in selected_techniques:
            st.write("Capturing frames...")
            start_recording = st.checkbox("Start Recording")
            if start_recording:
                background_subtraction_live()
            
    if st.checkbox("qualitative analysis for recorded dataset"):
        st.write("You selected option 2.")

        selected_techniques = st.multiselect("Select image processing techniques:",
                                                ["Optical Flow Algorithm", "Image Subtraction"])
        
        if "Image Subtraction" in selected_techniques:
            st.subheader("Image Subtraction technique")
            st.write("Upload a video file for background subtraction")
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

            if uploaded_file is not None:
                # Save the uploaded video file
                with open("temp_video.mov", "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Perform background subtraction on the uploaded video
                background_subtraction("temp_video.mov")

                # Show the processed video
                #st.video("stable_Background.avi")
                # Add a download button for the processed video
                st.download_button(
                    label="Download Processed Video",
                    data=open("stable_Background.avi", "rb").read(),
                    file_name="stable_Background.avi",
                    mime="video/avi")
        if "Optical Flow Algorithm" in selected_techniques:
            st.subheader("calculating optical flow")
            st.write("Upload a video file")
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

            if uploaded_file is not None:
                # Save the uploaded video file
                with open("temp_video.mov", "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Perform background subtraction on the uploaded video
                optical_flow("temp_video.mov")

                # Show the processed video
                #st.video("stable_Background.avi")
                # Add a download button for the processed video
                st.download_button(
                    label="Download Processed Video",
                    data=open("stable_Background.avi", "rb").read(),
                    file_name="stable_Background.avi",
                    mime="video/avi")
            
if __name__ == "__main__":
    main()
