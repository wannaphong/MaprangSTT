
import { NeedleError } from "../errors.js";
import { SPEECH_LANGUAGES, prepareAudio } from "./audio.js";
//#region src/runtime/engine.ts
/** Validate the documented Needle 3 container before handing it to native WASM code. */
function inspectModel(data) {
	const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
	const invalid = (detail) => {
		throw new NeedleError("INVALID_MODEL", `Invalid Needle 3 .cact model: ${detail}`);
	};
	if (data.length < 196 || view.getUint32(0, true) !== 98642564) invalid("wrong format tag or truncated header.");
	const count = view.getUint32(4, true);
	const directory = 196 + view.getUint32(8, true) * 4;
	if (!count || directory + count * 44 > data.length) invalid("truncated tensor directory.");
	for (let index = 0; index < count; index++) {
		const record = directory + index * 44;
		if (view.getBigUint64(record + 20, true) + view.getBigUint64(record + 28, true) > BigInt(data.length)) invalid(`tensor ${index} extends beyond the file.`);
	}
	if (data[directory + (count - 1) * 44] !== 4) return { calibrated: false };
	for (let heads = 1; heads <= 3; heads++) {
		const index = count - 2 - 6 * heads;
		if (index < 0) continue;
		const record = directory + index * 44;
		if (data[record] !== 1 || data[record + 1] !== 1 || view.getUint32(record + 4, true) !== heads) continue;
		const offset = Number(view.getBigUint64(record + 20, true));
		if (view.getBigUint64(record + 28, true) < BigInt(heads * 2)) invalid("truncated head manifest.");
		for (let head = 0; head < heads; head++) if (view.getUint16(offset + head * 2, true) === 16384) return { calibrated: true };
		return { calibrated: false };
	}
	return { calibrated: false };
}
function isCall(value) {
	if (!value || typeof value !== "object") return false;
	const call = value;
	return typeof call.name === "string" && call.arguments !== null && typeof call.arguments === "object" && !Array.isArray(call.arguments);
}
function nonnegative(value) {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
function validWords(words) {
	return Array.isArray(words) && words.every((word) => word && typeof word.word === "string" && nonnegative(word.start) && nonnegative(word.end) && word.end >= word.start && nonnegative(word.probability) && word.probability <= 1);
}
var Engine = class Engine {
	module;
	config;
	calibrated;
	#output;
	#toolsJson;
	constructor(module, config, calibrated) {
		this.module = module;
		this.config = config;
		this.calibrated = calibrated;
		this.#output = this.#allocate(config.bufferSize);
	}
	static create(module, config, data) {
		const { calibrated } = inspectModel(data);
		const engine = new Engine(module, config, calibrated);
		const model = engine.#allocate(data.length);
		module.HEAPU8.set(data, model);
		if (module._needle_load(model, BigInt(data.length)) < 0) throw engine.#error("needle_load");
		const speech = config.modelKind === "whistle";
		if (!(module._needle_models() & (speech ? 2 : 1))) throw new NeedleError("INVALID_MODEL", `Expected ${speech ? "Whistle speech" : "Needle text"} weights.`);
		if (!speech) engine.#configureTools(config.toolsJson);
		return engine;
	}
	#configureTools(toolsJson) {
		if (toolsJson === this.#toolsJson) return;
		this.#toolsJson = void 0;
		this.#withString(this.config.system, (system) => this.#withString(toolsJson, (tools) => {
			if (this.module._needle_init(system, tools, 0) < 0) throw this.#error("needle_init");
		}));
		this.#toolsJson = toolsJson;
	}
	#allocate(size) {
		const pointer = this.module._malloc(size);
		if (!pointer) throw new NeedleError("ENGINE_ERROR", `Unable to allocate ${size} bytes of WASM memory.`);
		return pointer;
	}
	#withString(value, run) {
		const bytes = new TextEncoder().encode(value);
		const pointer = this.#allocate(bytes.length + 1);
		try {
			this.module.HEAPU8.set(bytes, pointer);
			this.module.HEAPU8[pointer + bytes.length] = 0;
			return run(pointer);
		} finally {
			this.module._free(pointer);
		}
	}
	#error(operation) {
		const pointer = this.module._needle_last_error();
		const detail = pointer ? this.module.UTF8ToString(pointer) : "";
		return new NeedleError("ENGINE_ERROR", `${operation} failed${detail ? `: ${detail}` : "."}`);
	}
	#readJson() {
		const view = this.module.HEAPU8.subarray(this.#output, this.#output + this.config.bufferSize);
		const end = view.indexOf(0);
		if (end < 0) throw new NeedleError("INVALID_RESPONSE", "Output was truncated. Increase bufferSize.");
		try {
			return JSON.parse(new TextDecoder().decode(view.subarray(0, end)));
		} catch (cause) {
			throw new NeedleError("INVALID_RESPONSE", "Engine returned invalid JSON. Increase bufferSize.", { cause });
		}
	}
	complete(input, maxNewTokens, toolsJson = this.config.toolsJson) {
		this.#configureTools(toolsJson);
		if (this.config.stateless) this.reset();
		return this.#withString(input, (pointer) => {
			this.module.HEAPU8[this.#output] = 0;
			if (this.module._needle_complete(pointer, 0, 0, maxNewTokens, this.#output, this.config.bufferSize) < 0) throw this.#error("needle_complete");
			const result = this.#readJson();
			if (!result || typeof result !== "object" || typeof result.success !== "boolean" || ![
				"call",
				"respond",
				"refuse",
				"text"
			].includes(result.type) || !Array.isArray(result.function_calls) || !result.function_calls.every(isCall) || result.suppressed_calls !== void 0 && (!Array.isArray(result.suppressed_calls) || !result.suppressed_calls.every(isCall))) throw new NeedleError("INVALID_RESPONSE", "Engine returned an unexpected completion envelope.");
			result.suppressed_calls ??= [];
			if (!this.calibrated || typeof result.confidence !== "number" || !Number.isFinite(result.confidence)) result.confidence = null;
			if (result.peak_ram_mb !== void 0 && (typeof result.peak_ram_mb !== "number" || !Number.isFinite(result.peak_ram_mb) || result.peak_ram_mb < 0)) result.peak_ram_mb = null;
			return result;
		});
	}
	transcribe(audio, settings) {
		const { samples, durationInSeconds } = prepareAudio(audio);
		const pcm = this.#allocate(samples.byteLength);
		try {
			this.module.HEAPU8.set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength), pcm);
			return this.#withString(settings.language ?? "", (language) => this.#withString(settings.keywords?.join("\n") ?? "", (keywords) => {
				this.module.HEAPU8[this.#output] = 0;
				if (this.module._needle_transcribe(pcm, samples.length, settings.language ? language : 0, settings.keywords?.length ? keywords : 0, settings.wordTimestamps ? 1 : 0, this.#output, this.config.bufferSize) < 0) throw this.#error("needle_transcribe");
				const raw = this.#readJson();
				if (!raw || typeof raw !== "object" || typeof raw.text !== "string" || raw.language !== "" && !SPEECH_LANGUAGES.includes(raw.language) || !nonnegative(raw.ttft_ms) || !nonnegative(raw.decode_tps) || raw.words !== void 0 && !validWords(raw.words)) throw new NeedleError("INVALID_RESPONSE", "Engine returned an unexpected transcription envelope.");
				return {
					text: raw.text,
					language: raw.language,
					durationInSeconds,
					timeToFirstTokenMs: raw.ttft_ms,
					tokensPerSecond: raw.decode_tps,
					...raw.words === void 0 ? {} : { words: raw.words }
				};
			}));
		} finally {
			this.module._free(pcm);
		}
	}
	streamTranscribe(audio, settings) {
		this.module.HEAPU8[this.#output] = 0;
		let code;
		if (audio === void 0) code = this.module._needle_stream_transcribe_stop(this.#output, this.config.bufferSize);
		else {
			if (!(audio instanceof Float32Array)) throw new NeedleError("INVALID_ARGUMENT", "Stream chunks must be 16 kHz mono Float32Array samples.");
			const { samples } = prepareAudio(audio);
			const pcm = this.#allocate(samples.byteLength);
			try {
				this.module.HEAPU8.set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength), pcm);
				code = this.#withString(settings.language ?? "", (language) => this.#withString(settings.keywords?.join("\n") ?? "", (keywords) => this.module._needle_stream_transcribe_process(pcm, samples.length, settings.language ? language : 0, settings.keywords?.length ? keywords : 0, this.#output, this.config.bufferSize)));
			} finally {
				this.module._free(pcm);
			}
		}
		if (code < 0) throw this.#error("needle_stream_transcribe");
		const raw = this.#readJson();
		if (!raw || typeof raw !== "object" || typeof raw.text !== "string" || typeof raw.pending !== "string" || !validWords(raw.words) || raw.language !== "" && !SPEECH_LANGUAGES.includes(raw.language) || !nonnegative(raw.received) || !nonnegative(raw.pass_ms)) throw new NeedleError("INVALID_RESPONSE", "Engine returned an unexpected streaming transcription envelope.");
		return {
			text: raw.text,
			words: raw.words,
			pending: raw.pending,
			language: raw.language,
			received: raw.received,
			passMs: raw.pass_ms
		};
	}
	embed(input) {
		return this.#withString(input, (pointer) => this.#embedding(pointer, 0, 0));
	}
	embedAudio(audio) {
		const { samples } = prepareAudio(audio);
		const pcm = this.#allocate(samples.byteLength);
		try {
			this.module.HEAPU8.set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength), pcm);
			return this.#embedding(0, pcm, samples.length);
		} finally {
			this.module._free(pcm);
		}
	}
	#embedding(pointer, pcm, samples) {
		const dimension = this.module._needle_embed(pointer, pcm, samples, 0, 0);
		if (dimension <= 0) throw this.#error("needle_embed");
		if (dimension > 1048576) throw new NeedleError("INVALID_RESPONSE", `Unexpected embedding dimension: ${dimension}.`);
		const output = this.#allocate(dimension * 4);
		try {
			if (this.module._needle_embed(pointer, pcm, samples, output, dimension) !== dimension) throw this.#error("needle_embed");
			const embedding = new Float32Array(this.module.HEAPU8.buffer, output, dimension).slice();
			if (!embedding.every(Number.isFinite)) throw new NeedleError("INVALID_RESPONSE", "Engine returned a non-finite embedding.");
			return embedding;
		} finally {
			this.module._free(output);
		}
	}
	reset() {
		this.module._needle_reset();
	}
};
//#endregion
export { Engine, inspectModel };

//# sourceMappingURL=engine.js.map