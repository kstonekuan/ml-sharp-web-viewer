import type { TrajectoryType } from "./trajectory/types";
import { GaussianViewer } from "./viewer/GaussianViewer";

console.log("[main] Script loaded");

// Get DOM elements
const containerElement = document.getElementById("canvas-container");
const viewerFrameElement = document.getElementById("viewer-frame");
console.log("[main] Container element:", containerElement);
const fileLoaderElement = document.getElementById("file-loader");
const fileInputElement = document.getElementById(
	"file-input",
) as HTMLInputElement;
const trajectorySelectElement = document.getElementById(
	"trajectory-select",
) as HTMLSelectElement;
const playButtonElement = document.getElementById(
	"play-btn",
) as HTMLButtonElement;
const pauseButtonElement = document.getElementById(
	"pause-btn",
) as HTMLButtonElement;
const resetButtonElement = document.getElementById(
	"reset-btn",
) as HTMLButtonElement;
const loadingElement = document.getElementById("loading");

if (!containerElement) {
	throw new Error("Canvas container not found");
}

console.log("[main] Initializing GaussianViewer...");

// Initialize viewer
const viewer = new GaussianViewer({
	container: containerElement,
	onLoad: () => {
		console.log("[main] Splat loaded successfully");
		hideLoading();
		enableControls();
	},
	onError: (error) => {
		console.error("[main] Failed to load splat:", error);
		hideLoading();
		alert(`Failed to load PLY file: ${error.message}`);
	},
	onTrajectoryStateChange: (state) => {
		console.log("[main] Trajectory state changed:", state);
		updateButtonStates(state);
	},
	onFrameChange: (_frame, _total) => {
		// Could add a progress indicator here
	},
	onAspectRatioChange: (width, height) => {
		// Update viewer frame aspect ratio based on image metadata
		console.log("[main] Aspect ratio changed:", width, "x", height);
		if (viewerFrameElement) {
			viewerFrameElement.style.aspectRatio = `${width} / ${height}`;
			// Wait for DOM to update, then resize the renderer to match
			requestAnimationFrame(() => {
				viewer.resize();
			});
		}
	},
});

console.log("[main] GaussianViewer initialized");

// UI State Management
function showLoading(): void {
	loadingElement?.classList.add("visible");
}

function hideLoading(): void {
	loadingElement?.classList.remove("visible");
}

function enableControls(): void {
	trajectorySelectElement.disabled = false;
	playButtonElement.disabled = false;
	pauseButtonElement.disabled = false;
	resetButtonElement.disabled = false;
}

function disableControls(): void {
	trajectorySelectElement.disabled = true;
	playButtonElement.disabled = true;
	pauseButtonElement.disabled = true;
	resetButtonElement.disabled = true;
}

function updateButtonStates(state: "stopped" | "playing" | "paused"): void {
	playButtonElement.disabled = state === "playing";
	pauseButtonElement.disabled = state !== "playing";
}

// File Loading
async function loadFile(file: File): Promise<void> {
	console.log("[main] loadFile called with:", file.name, file.size, "bytes");

	if (!file.name.toLowerCase().endsWith(".ply")) {
		alert("Please select a PLY file");
		return;
	}

	showLoading();
	disableControls();

	try {
		console.log("[main] Calling viewer.loadPly...");
		await viewer.loadPly(file);
		console.log("[main] viewer.loadPly completed");
	} catch (error) {
		console.error("[main] loadFile error:", error);
		// Error already handled in viewer's onError callback
	}
}

// Event Listeners
fileInputElement?.addEventListener("change", (event) => {
	const target = event.target as HTMLInputElement;
	const file = target.files?.[0];
	if (file) {
		loadFile(file);
	}
});

fileLoaderElement?.addEventListener("click", () => {
	fileInputElement?.click();
});

// Drag and drop
fileLoaderElement?.addEventListener("dragover", (event) => {
	event.preventDefault();
	fileLoaderElement.classList.add("drag-over");
});

fileLoaderElement?.addEventListener("dragleave", () => {
	fileLoaderElement.classList.remove("drag-over");
});

fileLoaderElement?.addEventListener("drop", (event) => {
	event.preventDefault();
	fileLoaderElement.classList.remove("drag-over");

	const file = event.dataTransfer?.files[0];
	if (file) {
		loadFile(file);
	}
});

// Trajectory controls
trajectorySelectElement?.addEventListener("change", () => {
	const type = trajectorySelectElement.value as TrajectoryType;
	viewer.setTrajectoryType(type);
});

playButtonElement?.addEventListener("click", () => {
	viewer.play();
});

pauseButtonElement?.addEventListener("click", () => {
	viewer.pause();
});

resetButtonElement?.addEventListener("click", () => {
	viewer.reset();
});

// Keyboard shortcuts
document.addEventListener("keydown", (event) => {
	if (!viewer.isLoaded()) return;

	switch (event.key) {
		case " ":
			event.preventDefault();
			if (viewer.getPlayerState() === "playing") {
				viewer.pause();
			} else {
				viewer.play();
			}
			break;
		case "r":
		case "R":
			viewer.reset();
			break;
		case "Escape":
			viewer.stop();
			break;
	}
});

// Check for URL parameter to auto-load a file
const urlParams = new URLSearchParams(window.location.search);
const fileUrl = urlParams.get("file");

if (fileUrl) {
	showLoading();
	fetch(fileUrl)
		.then((response) => {
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			return response.blob();
		})
		.then((blob) => {
			const file = new File([blob], "scene.ply", {
				type: "application/octet-stream",
			});
			return loadFile(file);
		})
		.catch((error) => {
			hideLoading();
			console.error("Failed to load file from URL:", error);
		});
}
