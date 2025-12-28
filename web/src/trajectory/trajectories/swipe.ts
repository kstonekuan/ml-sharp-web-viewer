import { Vector3 } from "three";
import type { MaxOffset } from "../types";

/**
 * Create a left to right swipe trajectory.
 * Port of create_eye_trajectory_swipe from camera.py lines 113-125
 */
export function createSwipeTrajectory(
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	const positions: Vector3[] = [];

	for (let i = 0; i < numSteps; i++) {
		const t = numSteps > 1 ? i / (numSteps - 1) : 0;
		const x = offset.x * (2 * t - 1); // linspace(-offset, +offset)
		positions.push(new Vector3(x, 0, distanceMeters));
	}

	// Repeat the trajectory
	const result: Vector3[] = [];
	for (let r = 0; r < numRepeats; r++) {
		for (const pos of positions) {
			result.push(pos.clone());
		}
	}

	return result;
}
