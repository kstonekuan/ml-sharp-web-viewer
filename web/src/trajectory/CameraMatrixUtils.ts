import { Matrix4, Vector3 } from "three";
import type { MaxOffset, TrajectoryParams } from "./types";

/**
 * Compute the maximum offset for camera along X/Y/Z axis.
 * Port of compute_max_offset from camera.py lines 53-71
 *
 * @param minDepth - Minimum depth of the scene (closest point)
 * @param resolutionPx - Image resolution [width, height]
 * @param focalLengthPx - Focal length in pixels
 * @param params - Trajectory parameters
 */
export function computeMaxOffset(
	minDepth: number,
	resolutionPx: [number, number],
	focalLengthPx: number,
	params: TrajectoryParams,
): MaxOffset {
	const [width, height] = resolutionPx;
	const diagonal = Math.sqrt(
		(width / focalLengthPx) ** 2 + (height / focalLengthPx) ** 2,
	);

	const maxLateralOffset = params.maxDisparity * diagonal * minDepth;
	const maxMedialOffset = params.maxZoom * minDepth;

	return {
		x: maxLateralOffset,
		y: maxLateralOffset,
		z: maxMedialOffset,
	};
}

/**
 * Create a look-at camera matrix.
 * Port of create_camera_matrix from camera.py lines 252-287
 *
 * Note: The Python version uses OpenCV convention (Y down, Z forward).
 * Three.js uses Y up, Z backward. We apply the appropriate transform.
 */
export function createCameraMatrix(
	position: Vector3,
	lookAtPosition: Vector3 = new Vector3(0, 0, 0),
	worldUp: Vector3 = new Vector3(0, 1, 0), // Three.js convention: Y up
): Matrix4 {
	const matrix = new Matrix4();
	matrix.lookAt(position, lookAtPosition, worldUp);
	matrix.setPosition(position);
	return matrix;
}

/**
 * Convert from OpenCV camera convention to Three.js.
 * OpenCV: X right, Y down, Z forward
 * Three.js: X right, Y up, Z backward
 *
 * This applies a 180-degree rotation around the X-axis.
 */
export function opencvToThreejs(position: Vector3): Vector3 {
	return new Vector3(position.x, -position.y, -position.z);
}

/**
 * Compute depth statistics from Gaussian positions.
 *
 * @param positions - Array of 3D positions from the splat
 * @returns min, median, and max depth values
 */
export function computeDepthQuantiles(
	positions: Float32Array,
	quantile = 0.1,
): { min: number; focus: number; max: number } {
	const depths: number[] = [];

	// Extract Z values (depth) from positions
	for (let i = 2; i < positions.length; i += 3) {
		const z = positions[i];
		if (z !== undefined && Number.isFinite(z) && z > 0) {
			depths.push(z);
		}
	}

	if (depths.length === 0) {
		return { min: 1, focus: 2, max: 10 };
	}

	depths.sort((a, b) => a - b);

	const minIndex = Math.floor(depths.length * quantile);
	const maxIndex = Math.floor(depths.length * (1 - quantile));
	const focusIndex = Math.floor(depths.length * 0.5);

	return {
		min: depths[minIndex] ?? depths[0] ?? 1,
		focus: depths[focusIndex] ?? depths[0] ?? 2,
		max: depths[maxIndex] ?? depths[depths.length - 1] ?? 10,
	};
}
