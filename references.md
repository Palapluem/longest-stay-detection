# References

- Ultralytics YOLO tracking documentation: https://docs.ultralytics.com/modes/track
- BoT-SORT: Robust Associations Multi-Pedestrian Tracking: https://arxiv.org/abs/2206.14651
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box: https://arxiv.org/abs/2110.06864
- HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking: https://arxiv.org/abs/2009.07736

## How These References Inform This Project

- Ultralytics provides the practical tracking API used by the implementation, including `model.track`, persistent IDs, BoT-SORT, and ByteTrack tracker configs.
- BoT-SORT supports the default tracker choice because the task has multiple pedestrians and possible occlusion.
- ByteTrack supports the fallback tracker choice because it is designed to reduce fragmented tracks from lower-confidence detections.
- HOTA is useful for future evaluation if labeled ground truth becomes available.