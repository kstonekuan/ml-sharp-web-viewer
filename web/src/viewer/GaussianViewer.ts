import { SplatMesh } from "@sparkjsdev/spark";
import { Color, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { computeMaxOffset } from "../trajectory/CameraMatrixUtils";
import { TrajectoryPlayer } from "../trajectory/TrajectoryPlayer";
import { createEyeTrajectory } from "../trajectory/trajectories";
import {
	DEFAULT_TRAJECTORY_PARAMS,
	type TrajectoryParams,
	type TrajectoryType,
} from "../trajectory/types";
import {
	estimateFocalLength,
	type PlyMetadata,
	parsePlyMetadata,
} from "../utils/plyMetadata";

export interface ViewerOptions {
	container: HTMLElement;
	onLoad?: () => void;
	onError?: (error: Error) => void;
	onTrajectoryStateChange?: (state: "stopped" | "playing" | "paused") => void;
	onFrameChange?: (frame: number, total: number) => void;
	/** Called when metadata is loaded with image dimensions for aspect ratio */
	onAspectRatioChange?: (width: number, height: number) => void;
}

export class GaussianViewer {
	private container: HTMLElement;
	private scene: Scene;
	private camera: PerspectiveCamera;
	private renderer: WebGLRenderer;
	private controls: OrbitControls;
	private splatMesh: SplatMesh | null = null;
	private trajectoryPlayer: TrajectoryPlayer;
	private trajectoryParams: TrajectoryParams;
	private metadata: PlyMetadata | null = null;
	private isDisposed = false;
	private animationFrameId: number | null = null;

	// Camera model state (matching Python's PinholeCameraModel)
	private lookAtTarget = new Vector3(0, 0, 0);
	private depthFocus = 2.0;
	private minDepth = 1.0;
	private focalLength = 512; // Computed focal length (from metadata or estimated)

	private options: ViewerOptions;

	constructor(options: ViewerOptions) {
		console.log("[GaussianViewer] Constructor called");
		this.options = options;
		this.container = options.container;

		console.log(
			"[GaussianViewer] Container size:",
			this.container.clientWidth,
			"x",
			this.container.clientHeight,
		);

		// Initialize Three.js scene
		this.scene = new Scene();
		this.scene.background = new Color(0x1a1a1a);
		console.log("[GaussianViewer] Scene created");

		// Initialize camera with OpenCV coordinate convention (Y-down, Z-forward)
		// This matches SHARP PLY files which use OpenCV convention
		const aspect = this.container.clientWidth / this.container.clientHeight;
		this.camera = new PerspectiveCamera(45, aspect, 0.01, 500);
		this.camera.position.set(0, 0, -3);
		this.camera.up.set(0, -1, 0); // OpenCV: Y-down
		console.log("[GaussianViewer] Camera created, aspect:", aspect);

		// Initialize renderer
		this.renderer = new WebGLRenderer({ antialias: true });
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setSize(
			this.container.clientWidth,
			this.container.clientHeight,
		);
		this.container.appendChild(this.renderer.domElement);
		console.log("[GaussianViewer] Renderer created and attached");

		// Initialize orbit controls with OpenCV up vector
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.05;
		this.controls.target.copy(this.lookAtTarget);
		// Set controls to use same up vector as camera (Y-down)
		this.controls.object.up.set(0, -1, 0);

		// Initialize trajectory player
		this.trajectoryPlayer = new TrajectoryPlayer(30);
		this.trajectoryPlayer.onStateChange = (state) => {
			this.options.onTrajectoryStateChange?.(state);

			// Re-enable controls when trajectory stops
			if (state !== "playing") {
				this.controls.enabled = true;
			}
		};
		this.trajectoryPlayer.onFrameChange = (frame, total) => {
			this.options.onFrameChange?.(frame, total);
		};

		// Default trajectory params
		this.trajectoryParams = { ...DEFAULT_TRAJECTORY_PARAMS };

		// Handle resize
		window.addEventListener("resize", this.handleResize);

		// Start render loop
		this.animate();
		console.log("[GaussianViewer] Constructor complete, render loop started");
	}

	private handleResize = (): void => {
		if (this.isDisposed) return;

		const width = this.container.clientWidth;
		const height = this.container.clientHeight;

		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(width, height);
	};

	private animate = (): void => {
		if (this.isDisposed) return;

		this.animationFrameId = requestAnimationFrame(this.animate);

		// Update trajectory if playing
		if (this.trajectoryPlayer.isPlaying()) {
			const eyePosition = this.trajectoryPlayer.update(performance.now());
			if (eyePosition) {
				// Apply camera position from trajectory
				// The trajectory gives us eye positions in OpenCV convention
				// Three.js uses Y-up, so we need to convert
				this.applyCameraFromEyePosition(eyePosition);
			}
		} else {
			// Update orbit controls when not playing trajectory
			this.controls.update();
		}

		this.renderer.render(this.scene, this.camera);
	};

	/**
	 * Apply camera position from eye position, matching Python's PinholeCameraModel.compute()
	 *
	 * Since we're using OpenCV coordinate convention (Y-down) for the camera,
	 * no coordinate transformation is needed.
	 *
	 * In Python:
	 * - eye_pos is the camera position
	 * - look_at_position is [0, 0, depth_focus]
	 * - world_up is [0, -1, 0] (Y points down)
	 */
	private applyCameraFromEyePosition(eyePosition: Vector3): void {
		// Eye positions from trajectory are in OpenCV convention
		// Camera looks at origin + depthFocus along Z axis
		const lookAt = new Vector3(0, 0, this.depthFocus);

		this.camera.position.copy(eyePosition);
		this.camera.lookAt(lookAt);
	}

	async loadPly(file: File): Promise<void> {
		console.log("[GaussianViewer] loadPly called with:", file.name);
		try {
			// Remove existing splat mesh
			if (this.splatMesh) {
				console.log("[GaussianViewer] Removing existing splat mesh");
				this.scene.remove(this.splatMesh);
				this.splatMesh.dispose();
				this.splatMesh = null;
			}

			// Read file and parse metadata
			console.log("[GaussianViewer] Reading file buffer...");
			const buffer = await file.arrayBuffer();
			console.log("[GaussianViewer] Buffer size:", buffer.byteLength);
			this.metadata = parsePlyMetadata(buffer);
			console.log("[GaussianViewer] Metadata parsed:", this.metadata);

			// Create blob URL for Spark
			const blob = new Blob([buffer], { type: "application/octet-stream" });
			const url = URL.createObjectURL(blob);
			console.log("[GaussianViewer] Blob URL created:", url);

			// Load with Spark
			console.log("[GaussianViewer] Creating SplatMesh...");
			this.splatMesh = new SplatMesh({ url });
			this.scene.add(this.splatMesh);
			console.log("[GaussianViewer] SplatMesh added to scene");

			// Wait for load to complete
			console.log("[GaussianViewer] Waiting for SplatMesh to initialize...");
			await this.splatMesh.initialized;
			console.log("[GaussianViewer] SplatMesh initialized");

			URL.revokeObjectURL(url);

			// Compute depth quantiles and set up camera (matching Python)
			console.log("[GaussianViewer] Setting up camera for scene...");
			this.setupCameraForScene();

			// Generate initial trajectory
			console.log("[GaussianViewer] Generating trajectory...");
			this.generateTrajectory();

			console.log("[GaussianViewer] Load complete, calling onLoad");
			this.options.onLoad?.();
		} catch (error) {
			console.error("[GaussianViewer] loadPly error:", error);
			const err = error instanceof Error ? error : new Error(String(error));
			this.options.onError?.(err);
			throw err;
		}
	}

	/**
	 * Set up camera to view the splat.
	 * Matches Python's PinholeCameraModel behavior:
	 * - Camera starts at origin (0, 0, 0)
	 * - Camera looks at (0, 0, depth_focus)
	 * - depth_focus = max(2.0, 10th percentile of scene depths)
	 * - FOV computed from focal length and image height
	 */
	private setupCameraForScene(): void {
		if (!this.splatMesh) return;

		// Get bounding box using Spark's method
		const box = this.splatMesh.getBoundingBox(true);

		console.log("[GaussianViewer] Bounding box min:", box.min);
		console.log("[GaussianViewer] Bounding box max:", box.max);

		// Compute depth quantiles (matching Python's _compute_depth_quantiles)
		// In Python: depth_quantiles.focus = 10th percentile of scene Z values
		// Approximate from bounding box: focus ≈ min + 0.1 * (max - min)
		const minZ = box.min.z;
		const maxZ = box.max.z;
		this.minDepth = Math.max(0.1, minZ);
		// Python uses min_depth_focus=2.0 as floor
		this.depthFocus = Math.max(2.0, minZ + 0.1 * (maxZ - minZ));

		console.log("[GaussianViewer] Min depth:", this.minDepth);
		console.log("[GaussianViewer] Depth focus:", this.depthFocus);

		// Compute FOV from metadata focal length and image height
		// Python: fov = 2 * atan(height / (2 * focal_length))
		if (this.metadata) {
			const [imageWidth, imageHeight] = this.metadata.imageSize;
			// Use metadata focal length, or estimate from image size if not available
			this.focalLength =
				this.metadata.focalLength > 0
					? this.metadata.focalLength
					: estimateFocalLength(this.metadata.imageSize);

			console.log(
				"[GaussianViewer] Focal length:",
				this.focalLength,
				this.metadata.focalLength > 0 ? "(from metadata)" : "(estimated)",
			);

			// Compute vertical FOV in degrees
			const fovY =
				2 * Math.atan(imageHeight / (2 * this.focalLength)) * (180 / Math.PI);
			this.camera.fov = fovY;
			// Don't set camera.aspect here - it will be set by resize() after frame updates
			this.camera.updateProjectionMatrix();

			console.log("[GaussianViewer] FOV:", fovY);
			console.log("[GaussianViewer] Image size:", imageWidth, "x", imageHeight);

			// Notify about aspect ratio change for canvas resizing
			// The callback should call resize() after the DOM updates
			this.options.onAspectRatioChange?.(imageWidth, imageHeight);
		}

		// Camera starts at origin, looks at depth_focus along Z axis
		// This matches Python's eye_pos=[0,0,0] looking at [0,0,depth_focus]
		this.camera.position.set(0, 0, 0);
		this.lookAtTarget.set(0, 0, this.depthFocus);
		this.controls.target.set(0, 0, this.depthFocus);
		this.camera.lookAt(this.lookAtTarget);
		this.controls.update();

		console.log("[GaussianViewer] Camera position:", this.camera.position);
		console.log("[GaussianViewer] Look-at target:", this.lookAtTarget);
		console.log("[GaussianViewer] Camera setup complete");
	}

	private generateTrajectory(): void {
		const offset = computeMaxOffset(
			this.minDepth,
			this.metadata?.imageSize ?? [640, 480],
			this.focalLength,
			this.trajectoryParams,
		);

		console.log("[GaussianViewer] Trajectory offset:", offset);

		const positions = createEyeTrajectory(
			this.trajectoryParams.type,
			offset,
			this.trajectoryParams.distanceMeters,
			this.trajectoryParams.numSteps,
			this.trajectoryParams.numRepeats,
		);

		console.log(
			"[GaussianViewer] Generated",
			positions.length,
			"trajectory positions",
		);
		if (positions.length > 0) {
			console.log("[GaussianViewer] First position:", positions[0]);
			console.log(
				"[GaussianViewer] Last position:",
				positions[positions.length - 1],
			);
		}

		this.trajectoryPlayer.setTrajectory(positions);
	}

	setTrajectoryType(type: TrajectoryType): void {
		this.trajectoryParams.type = type;
		this.generateTrajectory();
	}

	play(): void {
		if (!this.splatMesh) return;
		this.controls.enabled = false;
		this.trajectoryPlayer.play();
	}

	pause(): void {
		this.trajectoryPlayer.pause();
		this.controls.enabled = true;
	}

	reset(): void {
		this.trajectoryPlayer.reset();
		if (!this.trajectoryPlayer.isPlaying()) {
			// Reset camera to initial position
			this.camera.position.set(0, 0, 0);
			this.camera.lookAt(this.lookAtTarget);
			this.controls.update();
		}
	}

	stop(): void {
		this.trajectoryPlayer.stop();
		this.controls.enabled = true;
	}

	getPlayerState(): "stopped" | "playing" | "paused" {
		return this.trajectoryPlayer.getState();
	}

	isLoaded(): boolean {
		return this.splatMesh !== null;
	}

	/** Manually trigger resize to sync camera/renderer with container size */
	resize(): void {
		this.handleResize();
	}

	dispose(): void {
		this.isDisposed = true;

		if (this.animationFrameId !== null) {
			cancelAnimationFrame(this.animationFrameId);
		}

		window.removeEventListener("resize", this.handleResize);

		if (this.splatMesh) {
			this.scene.remove(this.splatMesh);
			this.splatMesh.dispose();
		}

		this.controls.dispose();
		this.renderer.dispose();

		if (this.renderer.domElement.parentNode) {
			this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
		}
	}
}
