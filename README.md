# Experiment-Id-Switch-Reduction
This repo. includes all the data and code used for experiementing the id switch reduction in Multi Object Tracking (MOT), based on Human Pose Estimation which is associated with the Research Seminar paper.

Multiple Object Tracking (MOT) has witnessed remarkable advances in recent years. However, existing studies often suffer from high number of identity switches, particularly in scenarios with occlusions and complex backgrounds. To address this, we examine an approach that leverages human pose landmarks. By clustering pedestrians' id numbers assigned by DeepSORT tracker, based on their distance between their pose landmarks, we aim to reduce the number of identity switches associations. We conduct an experiment with various clustering algorithms, including k-means and hierarchical clustering. While k-means demonstrated the highest Rand Index score with ground truth, none of the algorithms achieved perfect separation of distinct identities. These results indicate the complexity of human pose variations and the limitations of current clustering algorithms in handling real world tracking scenarios. Future research could explore more advanced clustering techniques or integrate additional features to improve identity switch reduction
