"""Pose estimation service using MediaPipe."""

import contextlib
import io
import time
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from culora.core.exceptions import CuLoRAError
from culora.domain import CuLoRAConfig
from culora.domain.enums.pose import (
    ArmPosition,
    LegPosition,
    PoseCategory,
    PoseDynamism,
    PoseOrientation,
    PoseSymmetry,
)
from culora.domain.models.pose import (
    BatchPoseResult,
    PoseAnalysis,
    PoseAnalysisResult,
    PoseClassification,
    PoseCluster,
    PoseClusteringResult,
    PoseDiversityAnalysis,
    PoseLandmark,
    PoseSelectionCriteria,
    PoseSelectionResult,
    PoseSimilarity,
    PoseVector,
)


class PoseServiceError(CuLoRAError):
    """Pose service specific errors."""

    pass


class PoseDetectionError(PoseServiceError):
    """Pose detection specific errors."""

    pass


class PoseClassificationError(PoseServiceError):
    """Pose classification specific errors."""

    pass


class PoseClusteringError(PoseServiceError):
    """Pose clustering specific errors."""

    pass


class PoseService:
    """Service for pose estimation and analysis using MediaPipe."""

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize pose service with configuration."""
        self.config = config
        self.pose_config = config.pose

        # MediaPipe pose estimation (using Any to avoid typing issues)
        self._pose_estimator: Any | None = None
        self._mp_pose: Any = getattr(mp.solutions, "pose", None)
        self._mp_drawing: Any = getattr(mp.solutions, "drawing_utils", None)

        # Pose analysis cache
        self._pose_cache: dict[str, PoseAnalysis] = {}

    def _get_pose_estimator(self) -> Any:
        """Get or create MediaPipe pose estimator."""
        if self._pose_estimator is None:
            if self._mp_pose is None:
                raise ImportError("mediapipe.solutions.pose is not available")

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self._pose_estimator = self._mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.pose_config.model_complexity,
                    enable_segmentation=self.pose_config.enable_segmentation,
                    min_detection_confidence=self.pose_config.min_detection_confidence,
                    min_tracking_confidence=self.pose_config.min_tracking_confidence,
                )
        return self._pose_estimator

    def _prepare_image_for_pose(self, image: Image.Image) -> np.ndarray:
        """Prepare image for pose estimation."""
        # Resize if too large
        max_w, max_h = self.pose_config.max_image_size
        if image.size[0] > max_w or image.size[1] > max_h:
            image = image.copy()
            image.thumbnail((max_w, max_h), Image.LANCZOS)

        # Convert to RGB numpy array
        rgb_array = np.array(image.convert("RGB"))
        return rgb_array

    def _extract_landmarks(self, results: Any) -> list[PoseLandmark]:
        """Extract pose landmarks from MediaPipe results."""
        landmarks: list[PoseLandmark] = []

        if getattr(results, "pose_landmarks", None):
            for landmark in results.pose_landmarks.landmark:
                pose_landmark = PoseLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                    presence=getattr(landmark, "presence", landmark.visibility),
                )
                landmarks.append(pose_landmark)

        return landmarks

    def _create_pose_vector(self, landmarks: list[PoseLandmark]) -> PoseVector:
        """Create pose feature vector from landmarks."""
        if not landmarks:
            # Return empty vector for failed pose detection
            return PoseVector(
                vector=[0.0] * self.pose_config.feature_vector_dim,
                vector_dimension=self.pose_config.feature_vector_dim,
                confidence=0.0,
            )

        # Select key landmarks if configured
        if self.pose_config.key_landmarks_only:
            # Key landmarks: nose, shoulders, elbows, wrists, hips, knees, ankles
            key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            selected_landmarks = [
                landmarks[i] if i < len(landmarks) else landmarks[0]
                for i in key_indices
            ]
        else:
            selected_landmarks = landmarks

        # Create feature vector
        vector = []
        for landmark in selected_landmarks:
            if self.pose_config.normalize_coordinates:
                vector.extend([landmark.x, landmark.y])
            else:
                vector.extend([landmark.x, landmark.y])

            if self.pose_config.include_visibility:
                vector.append(landmark.visibility)

        # Pad or truncate to target dimension
        target_dim = self.pose_config.feature_vector_dim
        if len(vector) < target_dim:
            vector.extend([0.0] * (target_dim - len(vector)))
        elif len(vector) > target_dim:
            vector = vector[:target_dim]

        # Calculate confidence based on visible landmarks
        visible_count = sum(
            1
            for lm in selected_landmarks
            if lm.visibility > self.pose_config.min_landmark_confidence
        )
        confidence = (
            visible_count / len(selected_landmarks) if selected_landmarks else 0.0
        )

        return PoseVector(
            vector=vector,
            vector_dimension=len(vector),
            confidence=confidence,
        )

    def _classify_pose(self, landmarks: list[PoseLandmark]) -> PoseClassification:
        """Classify pose based on landmarks."""
        if not landmarks or len(landmarks) < 33:
            return PoseClassification(
                category=PoseCategory.UNKNOWN,
                orientation=PoseOrientation.UNKNOWN,
                arm_position=ArmPosition.UNKNOWN,
                leg_position=LegPosition.UNKNOWN,
                symmetry=PoseSymmetry.UNKNOWN,
                dynamism=PoseDynamism.UNKNOWN,
                confidence=0.0,
            )

        try:
            # Get key landmarks
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            # Pose category classification
            category = self._classify_pose_category(
                left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
            )

            # Orientation classification
            orientation = self._classify_orientation(
                nose, left_shoulder, right_shoulder
            )

            # Arm position classification
            arm_position = self._classify_arm_position(
                left_shoulder,
                right_shoulder,
                left_elbow,
                right_elbow,
                left_wrist,
                right_wrist,
                left_hip,
                right_hip,
            )

            # Leg position classification
            leg_position = self._classify_leg_position(
                left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
            )

            # Symmetry analysis
            symmetry = self._analyze_symmetry(landmarks)

            # Dynamism analysis
            dynamism = self._analyze_dynamism(landmarks)

            # Calculate overall confidence
            visible_landmarks = [
                lm
                for lm in landmarks
                if lm.visibility > self.pose_config.min_landmark_confidence
            ]
            confidence = len(visible_landmarks) / len(landmarks)

            return PoseClassification(
                category=category,
                orientation=orientation,
                arm_position=arm_position,
                leg_position=leg_position,
                symmetry=symmetry,
                dynamism=dynamism,
                confidence=confidence,
            )

        except Exception:
            return PoseClassification(
                category=PoseCategory.UNKNOWN,
                orientation=PoseOrientation.UNKNOWN,
                arm_position=ArmPosition.UNKNOWN,
                leg_position=LegPosition.UNKNOWN,
                symmetry=PoseSymmetry.UNKNOWN,
                dynamism=PoseDynamism.UNKNOWN,
                confidence=0.0,
            )

    def _classify_pose_category(
        self,
        left_hip: PoseLandmark,
        right_hip: PoseLandmark,
        left_knee: PoseLandmark,
        right_knee: PoseLandmark,
        left_ankle: PoseLandmark,
        right_ankle: PoseLandmark,
    ) -> PoseCategory:
        """Classify the main pose category."""
        # Calculate hip-knee-ankle angles to determine pose
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2

        # Standing: hips above knees above ankles
        if hip_y < knee_y < ankle_y:
            return PoseCategory.STANDING

        # Sitting: knees higher than ankles, hips similar to knees
        if knee_y < ankle_y and abs(hip_y - knee_y) < 0.1:
            return PoseCategory.SITTING

        # Lying: all joints at similar height
        if abs(hip_y - knee_y) < 0.05 and abs(knee_y - ankle_y) < 0.05:
            return PoseCategory.LYING

        # Kneeling: knees close to ankles, hips higher
        if abs(knee_y - ankle_y) < 0.1 and hip_y < knee_y:
            return PoseCategory.KNEELING

        # Crouching: all joints close together
        if abs(hip_y - ankle_y) < 0.2:
            return PoseCategory.CROUCHING

        return PoseCategory.UNKNOWN

    def _classify_orientation(
        self,
        nose: PoseLandmark,
        left_shoulder: PoseLandmark,
        right_shoulder: PoseLandmark,
    ) -> PoseOrientation:
        """Classify body orientation."""
        # Calculate shoulder distance to determine orientation
        shoulder_dist = abs(left_shoulder.x - right_shoulder.x)

        # Frontal: both shoulders visible and separated
        if shoulder_dist > 0.15:
            return PoseOrientation.FRONTAL

        # Profile: shoulders very close (one hidden)
        if shoulder_dist < 0.05:
            return PoseOrientation.PROFILE

        # Three-quarter: intermediate shoulder separation
        if 0.05 <= shoulder_dist <= 0.15:
            return PoseOrientation.THREE_QUARTER

        return PoseOrientation.UNKNOWN

    def _classify_arm_position(
        self,
        left_shoulder: PoseLandmark,
        right_shoulder: PoseLandmark,
        left_elbow: PoseLandmark,
        right_elbow: PoseLandmark,
        left_wrist: PoseLandmark,
        right_wrist: PoseLandmark,
        left_hip: PoseLandmark,
        right_hip: PoseLandmark,
    ) -> ArmPosition:
        """Classify arm positioning."""
        # Calculate arm positions relative to body
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        wrist_y = (left_wrist.y + right_wrist.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2

        # Raised: wrists above shoulders
        if wrist_y < shoulder_y - 0.1:
            return ArmPosition.RAISED

        # Extended: wrists far from body center
        body_center_x = (left_shoulder.x + right_shoulder.x) / 2
        wrist_center_x = (left_wrist.x + right_wrist.x) / 2
        if abs(wrist_center_x - body_center_x) > 0.2:
            return ArmPosition.EXTENDED

        # On hips: wrists at hip level
        if abs(wrist_y - hip_y) < 0.1:
            return ArmPosition.ON_HIPS

        # At sides: wrists below shoulders, close to body
        if wrist_y > shoulder_y and abs(wrist_center_x - body_center_x) < 0.1:
            return ArmPosition.AT_SIDES

        return ArmPosition.UNKNOWN

    def _classify_leg_position(
        self,
        left_hip: PoseLandmark,
        right_hip: PoseLandmark,
        left_knee: PoseLandmark,
        right_knee: PoseLandmark,
        left_ankle: PoseLandmark,
        right_ankle: PoseLandmark,
    ) -> LegPosition:
        """Classify leg positioning."""
        # Calculate hip-knee-ankle alignment
        left_leg_straight = abs(left_hip.x - left_ankle.x) < 0.05
        right_leg_straight = abs(right_hip.x - right_ankle.x) < 0.05

        # Straight: both legs aligned
        if left_leg_straight and right_leg_straight:
            return LegPosition.STRAIGHT

        # Crossed: ankles crossed relative to hips
        hip_spread = abs(left_hip.x - right_hip.x)
        ankle_spread = abs(left_ankle.x - right_ankle.x)
        if ankle_spread < hip_spread * 0.5:
            return LegPosition.CROSSED

        # Spread: ankles wider than hips
        if ankle_spread > hip_spread * 1.5:
            return LegPosition.SPREAD

        # One raised: significant height difference
        if abs(left_ankle.y - right_ankle.y) > 0.2:
            return LegPosition.ONE_RAISED

        # Bent: knees not aligned with hips-ankles
        if not left_leg_straight or not right_leg_straight:
            return LegPosition.BENT

        return LegPosition.UNKNOWN

    def _analyze_symmetry(self, landmarks: list[PoseLandmark]) -> PoseSymmetry:
        """Analyze pose symmetry."""
        try:
            # Compare left and right side landmarks
            symmetric_pairs = [
                (11, 12),  # shoulders
                (13, 14),  # elbows
                (15, 16),  # wrists
                (23, 24),  # hips
                (25, 26),  # knees
                (27, 28),  # ankles
            ]

            symmetry_scores = []
            for left_idx, right_idx in symmetric_pairs:
                if left_idx < len(landmarks) and right_idx < len(landmarks):
                    left_lm = landmarks[left_idx]
                    right_lm = landmarks[right_idx]

                    # Calculate position difference (mirrored)
                    y_diff = abs(left_lm.y - right_lm.y)
                    # For x, we expect mirror symmetry
                    center_x = 0.5  # Assuming normalized coordinates
                    left_dist = abs(left_lm.x - center_x)
                    right_dist = abs(right_lm.x - center_x)
                    x_diff = abs(left_dist - right_dist)

                    # Combine position differences
                    position_diff = (y_diff + x_diff) / 2
                    symmetry_scores.append(1.0 - min(position_diff * 5, 1.0))

            if symmetry_scores:
                avg_symmetry = sum(symmetry_scores) / len(symmetry_scores)
                return (
                    PoseSymmetry.SYMMETRIC
                    if avg_symmetry > 0.7
                    else PoseSymmetry.ASYMMETRIC
                )

        except Exception:
            pass

        return PoseSymmetry.UNKNOWN

    def _analyze_dynamism(self, landmarks: list[PoseLandmark]) -> PoseDynamism:
        """Analyze pose dynamism level."""
        try:
            # Check for action indicators
            # Large joint angles suggest dynamic poses
            if len(landmarks) >= 33:
                # Check elbow angles
                left_elbow_angle = self._calculate_angle(
                    landmarks[11], landmarks[13], landmarks[15]  # shoulder-elbow-wrist
                )
                right_elbow_angle = self._calculate_angle(
                    landmarks[12], landmarks[14], landmarks[16]
                )

                # Check knee angles
                left_knee_angle = self._calculate_angle(
                    landmarks[23], landmarks[25], landmarks[27]  # hip-knee-ankle
                )
                right_knee_angle = self._calculate_angle(
                    landmarks[24], landmarks[26], landmarks[28]
                )

                # Action: extreme joint angles
                if (
                    left_elbow_angle < 60
                    or right_elbow_angle < 60
                    or left_knee_angle < 90
                    or right_knee_angle < 90
                ):
                    return PoseDynamism.ACTION

                # Dynamic: moderate joint angles
                if (
                    left_elbow_angle < 120
                    or right_elbow_angle < 120
                    or left_knee_angle < 150
                    or right_knee_angle < 150
                ):
                    return PoseDynamism.DYNAMIC

                # Static: straight/relaxed pose
                return PoseDynamism.STATIC

        except Exception:
            pass

        return PoseDynamism.UNKNOWN

    def _calculate_angle(
        self, point1: PoseLandmark, point2: PoseLandmark, point3: PoseLandmark
    ) -> float:
        """Calculate angle between three points."""
        # Create vectors
        v1 = np.array([point1.x - point2.x, point1.y - point2.y])
        v2 = np.array([point3.x - point2.x, point3.y - point2.y])

        # Calculate angle
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle zero-length vectors to avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(np.degrees(angle))

    def _calculate_pose_bbox(
        self, landmarks: list[PoseLandmark]
    ) -> tuple[float, float, float, float]:
        """Calculate bounding box for pose landmarks."""
        if not landmarks:
            return (0.0, 0.0, 0.0, 0.0)

        visible_landmarks = [
            lm
            for lm in landmarks
            if lm.visibility > self.pose_config.min_landmark_confidence
        ]

        if not visible_landmarks:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = min(lm.x for lm in visible_landmarks)
        max_x = max(lm.x for lm in visible_landmarks)
        min_y = min(lm.y for lm in visible_landmarks)
        max_y = max(lm.y for lm in visible_landmarks)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _calculate_pose_score(
        self, landmarks: list[PoseLandmark], classification: PoseClassification
    ) -> float:
        """Calculate overall pose quality score."""
        if not landmarks:
            return 0.0

        # Visibility score
        visible_count = sum(
            1
            for lm in landmarks
            if lm.visibility > self.pose_config.min_landmark_confidence
        )
        visibility_score = visible_count / len(landmarks)

        # Classification confidence
        classification_score = classification.confidence

        # Completeness score (key landmarks present)
        key_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # Essential landmarks
        key_visible = sum(
            1
            for i in key_indices
            if i < len(landmarks)
            and landmarks[i].visibility > self.pose_config.min_landmark_confidence
        )
        completeness_score = key_visible / len(key_indices)

        # Combine scores
        return (
            visibility_score * 0.4
            + classification_score * 0.3
            + completeness_score * 0.3
        )

    def analyze_pose(self, image: Image.Image, image_path: Path) -> PoseAnalysisResult:
        """Analyze pose in a single image."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = str(image_path)
            if self.pose_config.enable_pose_cache and cache_key in self._pose_cache:
                cached_analysis = self._pose_cache[cache_key]
                return PoseAnalysisResult(
                    path=image_path,
                    success=True,
                    pose_analysis=cached_analysis,
                    analysis_duration=time.time() - start_time,
                )

            # Prepare image
            rgb_array = self._prepare_image_for_pose(image)

            # Perform pose estimation
            pose_estimator = self._get_pose_estimator()
            results = pose_estimator.process(rgb_array)

            # Extract landmarks
            landmarks = self._extract_landmarks(results)

            # Check minimum requirements
            if len(landmarks) < self.pose_config.min_visible_landmarks:
                return PoseAnalysisResult(
                    path=image_path,
                    success=False,
                    error="Insufficient visible landmarks detected",
                    error_code="INSUFFICIENT_LANDMARKS",
                    analysis_duration=time.time() - start_time,
                )

            # Create pose vector
            pose_vector = self._create_pose_vector(landmarks)

            # Classify pose
            classification = self._classify_pose(landmarks)

            # Calculate bounding box
            bbox = self._calculate_pose_bbox(landmarks)

            # Calculate pose score
            pose_score = self._calculate_pose_score(landmarks, classification)

            # Check minimum pose score
            if pose_score < self.pose_config.min_pose_score:
                return PoseAnalysisResult(
                    path=image_path,
                    success=False,
                    error=f"Pose score {pose_score:.3f} below minimum {self.pose_config.min_pose_score}",
                    error_code="LOW_POSE_SCORE",
                    analysis_duration=time.time() - start_time,
                )

            # Create pose analysis
            pose_analysis = PoseAnalysis(
                path=image_path,
                landmarks=landmarks,
                pose_vector=pose_vector,
                classification=classification,
                bbox=bbox,
                pose_score=pose_score,
                analysis_duration=time.time() - start_time,
            )

            # Cache result
            if self.pose_config.enable_pose_cache:
                self._pose_cache[cache_key] = pose_analysis

            return PoseAnalysisResult(
                path=image_path,
                success=True,
                pose_analysis=pose_analysis,
                analysis_duration=time.time() - start_time,
            )

        except Exception as e:
            return PoseAnalysisResult(
                path=image_path,
                success=False,
                error=str(e),
                error_code="POSE_ANALYSIS_FAILED",
                analysis_duration=time.time() - start_time,
            )

    def analyze_batch_poses(
        self, images_and_paths: list[tuple[Image.Image, Path]]
    ) -> BatchPoseResult:
        """Analyze poses in a batch of images."""
        start_time = time.time()
        results = []

        for image, path in images_and_paths:
            result = self.analyze_pose(image, path)
            results.append(result)

        # Calculate statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total_duration = time.time() - start_time
        poses_per_second = len(results) / total_duration if total_duration > 0 else 0

        # Calculate mean pose score
        if successful_results:
            pose_scores = [
                r.pose_analysis.pose_score
                for r in successful_results
                if r.pose_analysis
            ]
            mean_pose_score = (
                sum(pose_scores) / len(pose_scores) if pose_scores else 0.0
            )
        else:
            mean_pose_score = 0.0

        # Calculate pose statistics
        pose_statistics = {}
        if successful_results:
            all_vectors = [
                r.pose_analysis.pose_vector.vector
                for r in successful_results
                if r.pose_analysis
            ]
            if all_vectors:
                vector_array = np.array(all_vectors)
                pose_statistics = {
                    "mean_confidence": float(
                        np.mean(
                            [
                                r.pose_analysis.pose_vector.confidence
                                for r in successful_results
                                if r.pose_analysis
                            ]
                        )
                    ),
                    "vector_std": float(np.std(vector_array)),
                    "vector_mean": float(np.mean(vector_array)),
                }

        return BatchPoseResult(
            results=results,
            successful_analyses=len(successful_results),
            failed_analyses=len(failed_results),
            total_duration=total_duration,
            poses_per_second=poses_per_second,
            mean_pose_score=mean_pose_score,
            pose_statistics=pose_statistics,
        )

    def calculate_pose_similarity(
        self, pose1: PoseAnalysis, pose2: PoseAnalysis
    ) -> PoseSimilarity:
        """Calculate similarity between two poses."""
        # Calculate vector similarity using cosine similarity
        vector1 = np.array(pose1.pose_vector.vector)
        vector2 = np.array(pose2.pose_vector.vector)

        # Cosine similarity
        cos_sim = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        similarity_score = (cos_sim + 1) / 2  # Normalize to 0-1

        # Calculate distance
        distance = 1.0 - similarity_score

        # Calculate individual landmark similarities
        landmark_matches = {}
        if pose1.landmarks and pose2.landmarks:
            min_landmarks = min(len(pose1.landmarks), len(pose2.landmarks))
            for i in range(min_landmarks):
                lm1 = pose1.landmarks[i]
                lm2 = pose2.landmarks[i]

                # Calculate landmark distance
                lm_dist = np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)
                lm_similarity = max(
                    0.0, 1.0 - lm_dist * 2
                )  # Scale distance to similarity
                landmark_matches[f"landmark_{i}"] = lm_similarity

        return PoseSimilarity(
            path1=pose1.path,
            path2=pose2.path,
            similarity_score=similarity_score,
            distance=distance,
            landmark_matches=landmark_matches,
        )

    def analyze_pose_diversity(
        self, poses: list[PoseAnalysis]
    ) -> PoseDiversityAnalysis:
        """Analyze pose diversity in a dataset."""
        if len(poses) < 2:
            raise PoseServiceError("Need at least 2 poses for diversity analysis")

        # Calculate category distributions
        category_counts: dict[str, int] = {}
        orientation_counts: dict[str, int] = {}

        for pose in poses:
            cat = pose.classification.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

            orient = pose.classification.orientation
            orientation_counts[orient] = orientation_counts.get(orient, 0) + 1

        # Calculate pairwise similarities
        similarities = []
        most_similar_pairs: list[PoseSimilarity] = []
        most_diverse_pairs: list[PoseSimilarity] = []

        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                similarity = self.calculate_pose_similarity(poses[i], poses[j])
                similarities.append(similarity.similarity_score)

                # Track extreme pairs
                if len(most_similar_pairs) < self.pose_config.max_similarity_pairs:
                    most_similar_pairs.append(similarity)
                elif similarity.similarity_score > min(
                    p.similarity_score for p in most_similar_pairs
                ):
                    # Replace least similar in most similar list
                    min_idx = min(
                        range(len(most_similar_pairs)),
                        key=lambda x: most_similar_pairs[x].similarity_score,
                    )
                    most_similar_pairs[min_idx] = similarity

                if len(most_diverse_pairs) < self.pose_config.max_similarity_pairs:
                    most_diverse_pairs.append(similarity)
                elif similarity.similarity_score < max(
                    p.similarity_score for p in most_diverse_pairs
                ):
                    # Replace most similar in most diverse list
                    max_idx = max(
                        range(len(most_diverse_pairs)),
                        key=lambda x: most_diverse_pairs[x].similarity_score,
                    )
                    most_diverse_pairs[max_idx] = similarity

        # Sort pairs
        most_similar_pairs.sort(key=lambda x: x.similarity_score, reverse=True)
        most_diverse_pairs.sort(key=lambda x: x.similarity_score)

        # Calculate statistics
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        diversity_score = 1.0 - mean_similarity

        # Calculate similarity distribution
        similarities_array = np.array(similarities)
        similarity_distribution = {
            "p10": float(np.percentile(similarities_array, 10)),
            "p25": float(np.percentile(similarities_array, 25)),
            "p50": float(np.percentile(similarities_array, 50)),
            "p75": float(np.percentile(similarities_array, 75)),
            "p90": float(np.percentile(similarities_array, 90)),
        }

        return PoseDiversityAnalysis(
            total_images=len(poses),
            category_distribution=category_counts,
            orientation_distribution=orientation_counts,
            mean_pairwise_similarity=mean_similarity,
            diversity_score=diversity_score,
            similarity_distribution=similarity_distribution,
            most_similar_pairs=most_similar_pairs,
            most_diverse_pairs=most_diverse_pairs,
        )

    def cluster_poses(self, poses: list[PoseAnalysis]) -> PoseClusteringResult:
        """Cluster poses based on similarity."""
        if len(poses) < 2:
            raise PoseClusteringError("Need at least 2 poses for clustering")

        start_time = time.time()

        # Extract pose vectors
        vectors = [pose.pose_vector.vector for pose in poses]
        vector_array = np.array(vectors)

        # Determine optimal number of clusters
        max_clusters = min(self.pose_config.max_clusters, len(poses) // 2)
        if self.pose_config.enable_auto_clustering:
            optimal_k = self._determine_optimal_pose_clusters(
                vector_array, max_clusters
            )
        else:
            optimal_k = min(max_clusters, 5)  # Default

        # Perform clustering
        try:
            clustering = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
            cluster_labels = clustering.fit_predict(vector_array)
        except Exception as e:
            raise PoseClusteringError(f"Clustering failed: {e}") from e

        # Calculate silhouette score
        try:
            silhouette = silhouette_score(vector_array, cluster_labels)
        except Exception:
            silhouette = 0.0

        # Create clusters
        clusters = []
        cluster_size_dist = {}
        category_dist: dict[str, int] = {}

        for cluster_id in range(optimal_k):
            cluster_indices = [
                i for i, label in enumerate(cluster_labels) if label == cluster_id
            ]

            if len(cluster_indices) < self.pose_config.min_cluster_size:
                continue

            cluster_poses = [poses[i] for i in cluster_indices]
            cluster_paths = [pose.path for pose in cluster_poses]

            # Calculate centroid
            cluster_vectors = [vectors[i] for i in cluster_indices]
            centroid = np.mean(cluster_vectors, axis=0).tolist()

            # Calculate intra-cluster similarity
            if len(cluster_poses) > 1:
                cluster_similarities = []
                for i in range(len(cluster_poses)):
                    for j in range(i + 1, len(cluster_poses)):
                        sim = self.calculate_pose_similarity(
                            cluster_poses[i], cluster_poses[j]
                        )
                        cluster_similarities.append(sim.similarity_score)
                intra_similarity = sum(cluster_similarities) / len(cluster_similarities)
            else:
                intra_similarity = 1.0

            # Find dominant category
            cluster_categories = [
                pose.classification.category for pose in cluster_poses
            ]
            category_counts: dict[PoseCategory, int] = {}
            for cat in cluster_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            dominant_category = max(
                category_counts.keys(), key=lambda x: category_counts[x]
            )

            cluster = PoseCluster(
                cluster_id=cluster_id,
                image_paths=cluster_paths,
                centroid_vector=centroid,
                intra_cluster_similarity=intra_similarity,
                dominant_category=dominant_category,
                size=len(cluster_poses),
            )
            clusters.append(cluster)

            # Update distributions
            cluster_size_dist[f"cluster_{cluster_id}"] = len(cluster_poses)
            category_dist[dominant_category] = (
                category_dist.get(dominant_category, 0) + 1
            )

        processing_time = time.time() - start_time

        return PoseClusteringResult(
            num_clusters=len(clusters),
            clusters=clusters,
            silhouette_score=float(silhouette),
            cluster_size_distribution=cluster_size_dist,
            category_distribution=category_dist,
            processing_time=processing_time,
        )

    def _determine_optimal_pose_clusters(
        self, vectors: np.ndarray, max_clusters: int
    ) -> int:
        """Determine optimal number of clusters using elbow method."""
        if len(vectors) < 4:
            return 2

        inertias = []
        k_range = range(2, min(max_clusters + 1, len(vectors)))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                kmeans.fit(vectors)
                inertias.append(kmeans.inertia_)
            except Exception:
                break

        if len(inertias) < 2:
            return 2

        # Find elbow using difference method
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        if diffs:
            elbow_idx = diffs.index(max(diffs))
            return k_range[elbow_idx]

        return k_range[len(inertias) // 2]  # Middle value as fallback

    def select_diverse_poses(
        self,
        poses: list[PoseAnalysis],
        criteria: PoseSelectionCriteria,
        quality_scores: dict[Path, float] | None = None,
    ) -> PoseSelectionResult:
        """Select diverse poses based on criteria."""
        start_time = time.time()

        if len(poses) <= criteria.target_count:
            # Return all poses if we have fewer than target
            return PoseSelectionResult(
                selected_paths=[pose.path for pose in poses],
                selection_criteria=criteria,
                diversity_score=1.0,
                mean_quality_score=1.0,
                category_representation={},
                orientation_representation={},
                cluster_representation={},
                selection_reasoning=["All available poses selected"],
                processing_time=time.time() - start_time,
            )

        # Cluster poses for diversity selection
        clustering_result = self.cluster_poses(poses)

        # Initialize selection
        selected_poses = []
        selection_reasoning = []

        # Balance across clusters
        poses_per_cluster = max(
            1, criteria.target_count // len(clustering_result.clusters)
        )
        remaining_slots = criteria.target_count

        for cluster in clustering_result.clusters:
            cluster_poses = [p for p in poses if p.path in cluster.image_paths]

            # Determine how many to select from this cluster
            cluster_target = min(
                poses_per_cluster,
                len(cluster_poses),
                remaining_slots,
                criteria.max_cluster_representation or len(cluster_poses),
            )
            cluster_target = max(cluster_target, criteria.min_cluster_representation)

            if cluster_target > remaining_slots:
                cluster_target = remaining_slots

            # Select from cluster based on quality and diversity
            if quality_scores:
                # Sort by quality within cluster
                cluster_poses.sort(
                    key=lambda x: quality_scores.get(x.path, 0.5), reverse=True
                )

            # Select top poses from cluster
            selected_from_cluster = cluster_poses[:cluster_target]
            selected_poses.extend(selected_from_cluster)
            remaining_slots -= len(selected_from_cluster)

            selection_reasoning.append(
                f"Selected {len(selected_from_cluster)} poses from cluster {cluster.cluster_id} "
                f"({cluster.dominant_category})"
            )

            if remaining_slots <= 0:
                break

        # Fill remaining slots with best remaining poses
        if remaining_slots > 0:
            remaining_poses = [p for p in poses if p not in selected_poses]
            if quality_scores:
                remaining_poses.sort(
                    key=lambda x: quality_scores.get(x.path, 0.5), reverse=True
                )

            additional_selected = remaining_poses[:remaining_slots]
            selected_poses.extend(additional_selected)

            if additional_selected:
                selection_reasoning.append(
                    f"Added {len(additional_selected)} additional poses based on quality"
                )

        # Calculate result statistics
        selected_paths = [pose.path for pose in selected_poses]

        # Diversity score
        if len(selected_poses) > 1:
            diversity_analysis = self.analyze_pose_diversity(selected_poses)
            diversity_score = diversity_analysis.diversity_score
        else:
            diversity_score = 1.0

        # Quality score
        if quality_scores:
            quality_vals = [quality_scores.get(path, 0.5) for path in selected_paths]
            mean_quality = (
                sum(quality_vals) / len(quality_vals) if quality_vals else 0.5
            )
        else:
            mean_quality = 0.5

        # Category representation
        category_counts: dict[str, int] = {}
        orientation_counts: dict[str, int] = {}
        for pose in selected_poses:
            cat = pose.classification.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

            orient = pose.classification.orientation
            orientation_counts[orient] = orientation_counts.get(orient, 0) + 1

        # Cluster representation
        cluster_counts = {}
        for cluster in clustering_result.clusters:
            cluster_selected = [
                p for p in selected_poses if p.path in cluster.image_paths
            ]
            if cluster_selected:
                cluster_counts[cluster.cluster_id] = len(cluster_selected)

        return PoseSelectionResult(
            selected_paths=selected_paths,
            selection_criteria=criteria,
            diversity_score=diversity_score,
            mean_quality_score=mean_quality,
            category_representation=category_counts,
            orientation_representation=orientation_counts,
            cluster_representation=cluster_counts,
            selection_reasoning=selection_reasoning,
            processing_time=time.time() - start_time,
        )


def get_pose_service(config: CuLoRAConfig | None = None) -> PoseService:
    """Get or create pose service instance."""
    # In a real application, this might implement singleton pattern
    # For now, create new instance each time
    if config is None:
        from culora.services.config_service import get_config_service

        config = get_config_service().get_config()

    return PoseService(config)
