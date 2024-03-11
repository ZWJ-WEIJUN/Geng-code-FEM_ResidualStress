import cv2



from moviepy.editor import VideoFileClip
import cv2
from moviepy.editor import VideoFileClip, vfx


# ************************************************************Code for speeding up and cutting a video file************************************************************
clip = VideoFileClip("/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/STW70_Fr_Ctrl_run20240204.avi")
clip_speedup = clip.fx(vfx.speedx, factor=5)  # Adjust the speed factor as needed
# clip_speedup.write_videofile("/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/STW70_Fr_Ctrl_run20240204_SpeedUp.mp4")

# Cut the first 15 seconds of the video
clip_cut = clip_speedup.subclip(15)
clip_cut.write_videofile("/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/STW70_Fr_Ctrl_run20240204_SpeedUp_Cut15s.mp4")
# ************************************************************Code for speeding up and cutting a video file************************************************************


# ************************************************************Code for extracting frames from a video file************************************************************
def extract_frames(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    # Read and save each frame of the video
    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame was not read successfully, exit the loop
        if not ret:
            break

        # Save the frame as an image file
        frame_path = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video file
    video.release()

    print(f"Frames extracted: {frame_count}")

# # Example usage
# video_path = "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/STW70_LP_CTRL_run20240204.avi"
# output_folder = "/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/STW70_LP_CTRL_run20240204_Frames"
# extract_frames(video_path, output_folder)
    
# ************************************************************Code for extracting frames from a video file************************************************************
