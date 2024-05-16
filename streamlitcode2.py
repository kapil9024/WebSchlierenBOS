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
        ret, img = cap.read()
        if not ret:
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()



def main():
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

if __name__ == "__main__":
    main()
