// One Pyodide instance running py/explorer.py (a module worker). Messages in: {type: "preview" | "score" | "normals", id, params, req}.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

let py, ex;
const ready = (async () => {
  py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "scipy", "scikit-image"]);
  for (const f of ["global_curv.py", "explorer.py"]) {
    const src = await (await fetch("py/" + f, { cache: "no-cache" })).text();
    py.FS.writeFile(f, src);
  }
  py.runPython("import sys; sys.path.insert(0, '.')");
  ex = py.pyimport("explorer");
  self.postMessage({ type: "ready" });
})().catch((e) => self.postMessage({ type: "error", error: String(e) }));

async function ensureNormals(id) {
  if (ex.has_normals(id)) return;
  const r = await fetch(`data/rooms/normals/${id}.npz`);
  if (!r.ok) throw new Error(`normals ${id}: HTTP ${r.status}`);
  ex.add_normals(id, new Uint8Array(await r.arrayBuffer()));
}

function toJs(proxy) {
  const out = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return out;
}

self.onmessage = async ({ data: m }) => {
  await ready;
  try {
    const p = py.toPy(m.params || {});
    let result, transfer = [];
    if (m.type === "score") {
      if (!ex.has_chains(m.id, p)) await ensureNormals(m.id);
      result = toJs(ex.score(m.id, p));
    } else if (m.type === "preview") {
      if (!ex.has_chains(m.id, p)) await ensureNormals(m.id);
      result = toJs(ex.preview(m.id, p));
      for (const k of ["edges", "pts", "offsets", "codes"]) result[k] = result[k].slice();   // own copies, safe to transfer
      transfer = ["edges", "pts", "offsets", "codes"].map((k) => result[k].buffer);
    } else if (m.type === "normals") {
      await ensureNormals(m.id);
      const b = ex.normals_rgba(m.id);
      result = { rgba: b.toJs().slice() }; b.destroy();
      transfer = [result.rgba.buffer];
    }
    p.destroy();
    self.postMessage({ type: m.type, id: m.id, req: m.req, result }, transfer);
  } catch (e) {
    self.postMessage({ type: m.type, id: m.id, req: m.req, error: String(e) });
  }
};
