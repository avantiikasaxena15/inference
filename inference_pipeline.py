import cv2
import torch

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def process_video(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file at path: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {input_video_path}")
    print(f"Saving output to: {output_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results.render()[0]
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Output video saved at: {output_video_path}")

def main():
    input_video_path = '/workspaces/inference/test_videos.txt/video.mp4'  # Update this path
    output_video_path = '/workspaces/inference/output_video.mp4'

    model = load_model()
    process_video(input_video_path, output_video_path, model)

if __name__ == "__main__":
    main()
