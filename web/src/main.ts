import {
	DEFAULT_TRAJECTORY_PARAMS,
	type TrajectoryType,
} from "./trajectory/types";
import { GaussianViewer } from "./viewer/GaussianViewer";

console.log("[main] Script loaded");

// Get DOM elements
const containerElement = document.getElementById("canvas-container");
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
const loadSampleButtonElement = document.getElementById(
	"load-sample-btn",
) as HTMLButtonElement;

// Advanced settings elements
const advancedToggleElement = document.getElementById(
	"advanced-toggle",
) as HTMLButtonElement;
const advancedPanelElement = document.getElementById(
	"advanced-panel",
) as HTMLDivElement;
const maxDisparityInputElement = document.getElementById(
	"max-disparity-input",
) as HTMLInputElement;
const maxZoomInputElement = document.getElementById(
	"max-zoom-input",
) as HTMLInputElement;
const distanceInputElement = document.getElementById(
	"distance-input",
) as HTMLInputElement;
const numStepsInputElement = document.getElementById(
	"num-steps-input",
) as HTMLInputElement;
const numRepeatsInputElement = document.getElementById(
	"num-repeats-input",
) as HTMLInputElement;
const resetParamsButtonElement = document.getElementById(
	"reset-params-btn",
) as HTMLButtonElement;

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
	// Canvas stays fixed size - splat renders with empty space around it as needed
});

console.log("[main] GaussianViewer initialized");

// UI State Management
function showLoading(): void {
	loadingElement?.classList.add("visible");
}

function hideLoading(): void {
	loadingElement?.classList.remove("visible");
}

function setParameterControlsDisabled(disabled: boolean): void {
	advancedToggleElement.disabled = disabled;
	maxDisparityInputElement.disabled = disabled;
	maxZoomInputElement.disabled = disabled;
	distanceInputElement.disabled = disabled;
	numStepsInputElement.disabled = disabled;
	numRepeatsInputElement.disabled = disabled;
	resetParamsButtonElement.disabled = disabled;
}

function enableControls(): void {
	trajectorySelectElement.disabled = false;
	playButtonElement.disabled = false;
	pauseButtonElement.disabled = false;
	resetButtonElement.disabled = false;
	setParameterControlsDisabled(false);
}

function disableControls(): void {
	trajectorySelectElement.disabled = true;
	playButtonElement.disabled = true;
	pauseButtonElement.disabled = true;
	resetButtonElement.disabled = true;
	setParameterControlsDisabled(true);
}

function updateButtonStates(state: "stopped" | "playing" | "paused"): void {
	playButtonElement.disabled = state === "playing";
	pauseButtonElement.disabled = state !== "playing";
	// Disable parameter controls during playback
	setParameterControlsDisabled(state === "playing");
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

// Sample File Loading
async function loadSampleFile(): Promise<void> {
	const sampleFileUrl = `${import.meta.env.BASE_URL}samples/sample.ply`;
	console.log("[main] Loading sample file from:", sampleFileUrl);

	showLoading();
	disableControls();

	try {
		const response = await fetch(sampleFileUrl);
		if (!response.ok) {
			throw new Error(`Failed to fetch sample: HTTP ${response.status}`);
		}
		const blob = await response.blob();
		const file = new File([blob], "sample.ply", {
			type: "application/octet-stream",
		});
		await viewer.loadPly(file);
	} catch (error) {
		hideLoading();
		console.error("[main] Failed to load sample file:", error);
		const errorMessage = error instanceof Error ? error.message : String(error);
		alert(`Failed to load sample file: ${errorMessage}`);
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

// Load sample button
loadSampleButtonElement?.addEventListener("click", () => {
	loadSampleFile();
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

// Advanced settings toggle
advancedToggleElement?.addEventListener("click", () => {
	const isExpanded = advancedToggleElement.classList.toggle("expanded");
	advancedToggleElement.setAttribute("aria-expanded", String(isExpanded));
	advancedPanelElement?.classList.toggle("collapsed", !isExpanded);
});

// Wire up parameter controls
maxDisparityInputElement?.addEventListener("input", () => {
	const value = Number.parseFloat(maxDisparityInputElement.value);
	if (!Number.isNaN(value)) {
		viewer.updateTrajectoryParam("maxDisparity", value);
	}
});

maxZoomInputElement?.addEventListener("input", () => {
	const value = Number.parseFloat(maxZoomInputElement.value);
	if (!Number.isNaN(value)) {
		viewer.updateTrajectoryParam("maxZoom", value);
	}
});

distanceInputElement?.addEventListener("input", () => {
	const value = Number.parseFloat(distanceInputElement.value);
	if (!Number.isNaN(value)) {
		viewer.updateTrajectoryParam("distanceMeters", value);
	}
});

numStepsInputElement?.addEventListener("input", () => {
	const value = Number.parseInt(numStepsInputElement.value, 10);
	if (!Number.isNaN(value)) {
		viewer.updateTrajectoryParam("numSteps", value);
	}
});

numRepeatsInputElement?.addEventListener("input", () => {
	const value = Number.parseInt(numRepeatsInputElement.value, 10);
	if (!Number.isNaN(value)) {
		viewer.updateTrajectoryParam("numRepeats", value);
	}
});

// Reset to defaults
function updateParameterInputsFromDefaults(): void {
	const defaults = DEFAULT_TRAJECTORY_PARAMS;
	maxDisparityInputElement.value = String(defaults.maxDisparity);
	maxZoomInputElement.value = String(defaults.maxZoom);
	distanceInputElement.value = String(defaults.distanceMeters);
	numStepsInputElement.value = String(defaults.numSteps);
	numRepeatsInputElement.value = String(defaults.numRepeats);
}

resetParamsButtonElement?.addEventListener("click", () => {
	viewer.resetTrajectoryParams();
	updateParameterInputsFromDefaults();
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
