import { Vector3 } from "three";
import type { MaxOffset } from "../types";

/**
 * Create a rotating trajectory with forward movement (dolly).
 * Port of create_eye_trajectory_rotate_forward from camera.py lines 183-204
 */
export function createRotateForwardTrajectory(
	offset: MaxOffset,
	distanceMeters: number,
	numSteps: number,
	numRepeats: number,
): Vector3[] {
	const numStepsTotal = numSteps * numRepeats;
	const positions: Vector3[] = [];

	for (let i = 0; i < numStepsTotal; i++) {
		const t = (i / numStepsTotal) * numRepeats;
		const x = offset.x * Math.sin(2 * Math.PI * t);
		const z =
			distanceMeters + (offset.z * (1.0 - Math.cos(2 * Math.PI * t))) / 2;
		positions.push(new Vector3(x, 0, z));
	}

	return positions;
}
