import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
	base: "/ml-sharp/",
	server: {
		port: 3000,
	},
	build: {
		outDir: "dist",
		sourcemap: true,
	},
	resolve: {
		alias: {
			"@": resolve(__dirname, "src"),
		},
	},
});
