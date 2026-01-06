@staticmethod
def get_click_video(video_path, max_points=20):
    """
    Select positive (left-click) and negative (right-click) points on the first frame of the video.

    Args:
        video_path: path to input video (.mp4)
        max_points: maximum total number of clicks (for safety)

    Returns:
        pos_points: list of (x, y) positive clicks
        neg_points: list of (x, y) negative clicks
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Cannot read the first frame from video.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ObjectUncertaintyTracker.get_click_image(frame_rgb, max_points=max_points)

def _init_models(self, sam_model, ground_model):
    sam_model, sam_image_model, sam_processor, sam_image_processor = self.initialize_sam2_video(sam_model, self.device, self.dtype)
    ground_model = MolmoVLLMClient() if ground_model == "molmo" else VLMGround(model_name="Qwen2.5VL")
    return sam_model, sam_image_model, sam_processor, sam_image_processor, ground_model

def seg_object_in_image(self, image, show=True):
    """
    Segment object in a single image using SAM2.

    Args:
        image: PIL.Image input image.
        pos_points: list of (x, y) positive clicks.
        neg_points: list of (x, y) negative clicks.
    """
    pos_points, neg_points = self.get_click_image(image)
    points = [[[[x, y] for (x, y) in pos_points + (neg_points or [])]]]
    labels = [[[1] * len(pos_points) + [0] * len(neg_points or [])]]

    inputs = self.image_processor(
        images=image, 
        input_points=points, 
        input_labels=labels, 
        return_tensors="pt",
    ).to(self.device)

    with torch.no_grad():
        outputs = self.image_model(**inputs)

    best_mask_idx = torch.argmax(outputs.iou_scores.squeeze())
    masks = self.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0][0]
    mask = (masks[best_mask_idx].numpy() > 0.5).astype(np.uint8)

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap="jet")
        plt.axis("off")
        plt.title("Segmented Mask Overlay")
        plt.show()
    return mask

def track_object_seg(self, video_path, text):
    """
    Track objects across a video using SAM2 (Segment Anything Model 2).

    Args:
        video_path: path or URL to a video file.
        pos_points: list of (x, y) positive clicks on the first frame.
        neg_points: list of (x, y) negative clicks.
        model_name: pretrained SAM2 model checkpoint.
        save_path: output video path.
        dtype: torch dtype for inference.

    Returns:
        frames_with_masks: list of PIL.Image frames with overlays.
        masks: list of numpy binary masks, one per frame.
    """
    video_name = os.path.basename(video_path)
    cached_mask_path = os.path.join(self.output_dir, f"{video_name}_{text}_masks.npy")
    save_path = os.path.join(self.output_dir, f"{video_name}_{text}_tracked_seg.mp4")

    if os.path.exists(cached_mask_path):
        print(f"Loading cached masks from {cached_mask_path}")
        masks = np.load(cached_mask_path)
        frames = load_video(video_path)[0]
        assert len(frames) == len(masks), "Mismatch between frames and cached masks."
        return [Image.fromarray(f) for f in frames], masks

    assert os.path.exists(video_path), f"Video not found at {video_path}"
    pos_points, neg_points = self.get_click_video(video_path)

    # --- Load video frames ---
    video_frames, _ = load_video(video_path)
    height, width = video_frames[0].shape[0], video_frames[0].shape[1]

    # --- Initialize video inference session ---
    inference_session = self.processor.init_video_session(
        video=video_frames,
        inference_device=self.device,
        dtype=self.dtype,
    )

    # --- Prepare clicks ---
    ann_frame_idx = 0
    ann_obj_id = 1
    points = [[[[x, y] for (x, y) in pos_points + (neg_points or [])]]]
    labels = [[[1] * len(pos_points) + [0] * len(neg_points or [])]]

    self.processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=points,
        input_labels=labels,
    )

    # --- Segment the first frame ---
    outputs = self.model(inference_session=inference_session, frame_idx=ann_frame_idx)
    video_res_masks = self.processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]
    print(f"[SAM2] Initialized segmentation on frame {ann_frame_idx}, shape={video_res_masks.shape}")

    # --- Prepare visualization output ---
    out = None
    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    color = np.array([0, 255, 0], dtype=np.uint8)

    masks = {}
    # --- Propagate through all frames ---
    print("[SAM2] Tracking across video...")
    for sam2_video_output in tqdm(self.model.propagate_in_video_iterator(inference_session)):
        frame_idx = sam2_video_output.frame_idx
        video_res_masks = self.processor.post_process_masks(
            [sam2_video_output.pred_masks],
            original_sizes=[
                [inference_session.video_height, inference_session.video_width]
            ],
            binarize=True,
        )[0]
        mask = video_res_masks.squeeze().cpu().numpy().astype(np.uint8)
        masks[frame_idx] = mask

        # Overlay mask on frame
        frame = video_frames[frame_idx]
        overlay = frame.copy()
        overlay[mask > 0] = 0.6 * overlay[mask > 0] + 0.4 * color
        out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"[SAM2] Tracked object through {len(masks)} frames")
    print(f"[SAM2] Saved result video to {save_path}")

    # Sort masks by frame index
    ordered_masks = [masks[i] for i in sorted(masks.keys())]
    np.save(cached_mask_path, np.array(ordered_masks))

    return [Image.fromarray(f) for f in video_frames], ordered_masks