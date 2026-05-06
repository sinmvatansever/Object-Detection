This project consists of applying a traditional computer vision approach for detecting
objects without relying on modern deep-learning techniques. The developed algorithm
focuses on feature matching to find and locate objects across multiple query images,
using only one reference image that contains four target objects referred as “train image”.
Key points are first detected using SIFT, matched using FLANN, and then filtered with
RANSAC to remove incorrect matches. A homography is estimated from these matches
and used to project the reference object’s bounding box onto each query image. The
detection output is visualised by showing the projected boundaries together with the
inlier feature points.
The results show that the method can successfully detect the objects in three query
images of increasing difficulty while meeting the project limitations. The findings
demonstrate that homography-based detection works well for objects that are flat or
nearly flat, but the method becomes less reliable when there are large changes in
viewpoint, lighting, occlusion, or when the object has weak visual features.
