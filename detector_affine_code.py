import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

class ObjectDetector:
    def __init__(self, template_files, template_names, template_colors, inlier_colors,
                 output_folder="detection_results/results_affine", match_subplot_folder="detection_results/match_subplots_filtered",
                 params=None):
        
        # ---------------------------
        # Folders
        # ---------------------------
        self.output_folder = output_folder
        self.match_subplot_folder = match_subplot_folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.match_subplot_folder, exist_ok=True)
        
        # ---------------------------
        # Templates
        # ---------------------------
        self.template_files = template_files
        self.template_names = template_names
        self.template_colors = template_colors
        self.inlier_colors = inlier_colors
        
        # ---------------------------
        # Parameters (Optimal for 3 templates. For other templates, adjust parameters)
        # ---------------------------
        default_params = {
            'minMatches': 10,
            'minInliers': 4,
            'matchRatio': 0.7,
            'matchDistThresh': 300,
            'maxRansacDist': 20.0,
            'maxNumTrials': 8000,
            'confidence': 0.995,
            'minBoxArea': 2000,
            'maxBoxFrac': 0.98
        }
        self.params = default_params if params is None else params
        
        # ---------------------------
        # SIFT + FLANN
        # ---------------------------
        self.sift = cv2.SIFT_create(nfeatures=8000)
        FLANN_INDEX_KDTREE = 1
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict()
        )
        
        # ---------------------------
        # Load templates
        # ---------------------------
        self.template_imgs = []
        self.template_gray = []
        self.template_kps = []
        self.template_desc = []
        self._load_templates()
    
    # ---------------------------
    # Template loading
    # ---------------------------
    def _load_templates(self):
        for f in self.template_files:
            img = cv2.imread(f)
            if img is None:
                print(f"[WARNING] Template not found: {f}")
                self.template_imgs.append(None)
                self.template_gray.append(None)
                self.template_kps.append(None)
                self.template_desc.append(None)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, desc = self.sift.detectAndCompute(gray, None)
            self.template_imgs.append(img)
            self.template_gray.append(gray)
            self.template_kps.append(kps)
            self.template_desc.append(desc)
    
    # ---------------------------
    # Detect objects in a query image
    # ---------------------------
    def detect_in_image(self, query_file):
        Qimg_color = cv2.imread(query_file)
        if Qimg_color is None:
            print(f"[WARNING] Query not found: {query_file}")
            return
        
        Qimg = cv2.cvtColor(Qimg_color, cv2.COLOR_BGR2RGB)
        Qgray = cv2.cvtColor(Qimg_color, cv2.COLOR_BGR2GRAY)
        Hq, Wq = Qgray.shape
        
        kpsQ, descQ = self.sift.detectAndCompute(Qgray, None)
        
        print("\n=====================================================")
        print(f"Processing query image: {query_file}")
        print("=====================================================\n")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(Qimg)
        ax.axis('off')
        ax.set_title(query_file)
        
        detected_match_images = []
        detected_titles = []
        
        # ---------------------------
        # Loop over templates
        # ---------------------------
        for idx, tname in enumerate(self.template_names):
            Timg = self.template_imgs[idx]
            kpsT = self.template_kps[idx]
            descT = self.template_desc[idx]
            
            if descT is None:
                continue
            
            # FLANN matching
            try:
                matches = self.flann.knnMatch(descT, descQ, k=2)
            except:
                continue
            
            good = [m for m, n in matches if m.distance < self.params['matchRatio']*n.distance 
                    and m.distance < self.params['matchDistThresh']]
            
            print(f"[{tname}] good matches = {len(good)}")
            if len(good) < self.params['minMatches']:
                print(f"[{tname}]  Not enough matches, skipping.\n")
                continue
            
            ptsT = np.float32([kpsT[m.queryIdx].pt for m in good])
            ptsQ = np.float32([kpsQ[m.trainIdx].pt for m in good])
            
            # RANSAC
            M, mask = cv2.estimateAffinePartial2D(
                ptsT, ptsQ,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.params['maxRansacDist'],
                maxIters=self.params['maxNumTrials'],
                confidence=self.params['confidence']
            )
            if M is None:
                print(f"[{tname}]  RANSAC failed.\n")
                continue
            
            mask = mask.ravel()
            inliers = int(np.sum(mask))
            print(f"[{tname}] inliers = {inliers}")
            if inliers < self.params['minInliers']:
                print(f"[{tname}]  Too few inliers, skipping.\n")
                continue
            
            # Bounding box
            hT, wT = self.template_gray[idx].shape
            box = np.float32([[0,0],[wT,0],[wT,hT],[0,hT]]).reshape(-1,1,2)
            new_box = cv2.transform(box, M).reshape(-1,2)
            box_list = [(f"{x:.5f}", f"{y:.5f}") for x, y in new_box]
            print(f"[{tname}] box coords = {box_list}\n")
            
            # Draw box
            poly = Polygon(new_box, closed=True, fill=False,
                           edgecolor=self.template_colors[idx], linewidth=2)
            ax.add_patch(poly)
            ax.text(float(new_box[0,0]), float(new_box[0,1]) - 10,
                    f"{tname} ({inliers} inliers)",
                    color='yellow', fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6))
            
            # Draw inliers
            inlier_ptsQ = ptsQ[mask.astype(bool)]
            ax.scatter(inlier_ptsQ[:,0], inlier_ptsQ[:,1],
                       s=10, c=self.inlier_colors[idx])
            
            # Match subplot
            match_img = cv2.drawMatches(
                Timg, kpsT,
                Qimg_color, kpsQ,
                good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            detected_match_images.append(match_img)
            detected_titles.append(f"{tname} ({len(good)} matches)")
        
        # Save detection figure
        out_path = os.path.join(self.output_folder, os.path.basename(query_file))
        plt.tight_layout()
        plt.savefig(out_path, dpi=250)
        plt.close()
        print(f"[SAVED] Detection image → {out_path}")
        
        # Save match subplots
        if len(detected_match_images) > 0:
            rows = len(detected_match_images)
            fig, axs = plt.subplots(rows, 1, figsize=(20, 6*rows))
            if rows == 1:
                axs = [axs]
            for i, (img, title) in enumerate(zip(detected_match_images, detected_titles)):
                axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[i].set_title(title, fontsize=16)
                axs[i].axis('off')
            plt.tight_layout()
            out2 = os.path.join(
                self.match_subplot_folder,
                f"filtered_matches_{os.path.basename(query_file)}.png"
            )
            plt.savefig(out2, dpi=250)
            plt.close()
            print(f"[SAVED] Match subplot → {out2}")
    

# ---------------------------
# Usage
template_files = [
    'dataset/img_airpod.jpg',
    'dataset/img_pallette.jpg',
    'dataset/img_controller.jpg',
    'dataset/img_vitaminc.jpg'
]

template_names = ['AIRPODS', 'ARTCLASS', 'CONTROLLER', 'VITAMIN']
template_colors = ['r', 'g', 'b', 'm']
inlier_colors   = ['orange', 'cyan', 'yellow', 'lime']

detector = ObjectDetector(template_files, template_names, template_colors, inlier_colors)

query_files = [
    'dataset/img_query_1.jpg',
    'dataset/img_query_2.jpg',
    'dataset/img_query_3.jpg'
]

for qf in query_files:
    detector.detect_in_image(qf)