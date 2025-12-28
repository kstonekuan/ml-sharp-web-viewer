import { Vector3 } from "three";
import type { MaxOffset } from "../types";

/**
 * Create a forward-only trajectory (dolly in/out).
 * Port of create_eye_trajectory_forward from camera.py lines 207-219
 */
export function createForwardTrajectory(
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	const positions: Vector3[] = [];

	for (let i = 0; i < numSteps; i++) {
		const t = numSteps > 1 ? i / (numSteps - 1) : 0;
		const z = distanceMeters + offset.z * t;
		positions.push(new Vector3(0, 0, z));
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
