// recordingWorker.js - Background worker for non-blocking recording
'use strict';

const CHUNK_SIZE = 100;

let db = null;
let buffer = [];
let chunks = [];
let frameCount = 0;

// Initialize IndexedDB
async function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('HelioRecordings', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('chunks')) {
        db.createObjectStore('chunks', { keyPath: 'id' });
      }
    };
  });
}

// Flush buffer to IndexedDB
async function flushChunk() {
  if (buffer.length === 0 || !db) return;

  const chunkId = chunks.length;
  const chunk = { id: chunkId, snapshots: buffer.slice() };

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['chunks'], 'readwrite');
    const store = transaction.objectStore('chunks');
    const request = store.add(chunk);
    request.onsuccess = () => {
      chunks.push(chunkId);
      buffer = [];
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}

// Clear all chunks
async function clearDB() {
  if (!db) return;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['chunks'], 'readwrite');
    const store = transaction.objectStore('chunks');
    const request = store.clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Retrieve all chunks
async function getAllSnapshots() {
  const allSnapshots = [];
  for (const chunkId of chunks) {
    const chunk = await new Promise((resolve, reject) => {
      const transaction = db.transaction(['chunks'], 'readonly');
      const store = transaction.objectStore('chunks');
      const request = store.get(chunkId);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    if (chunk && chunk.snapshots) {
      allSnapshots.push(...chunk.snapshots);
    }
  }
  return allSnapshots;
}

// Message handler
self.onmessage = async (e) => {
  const { type, payload } = e.data;

  try {
    switch (type) {
      case 'init':
        db = await initDB();
        await clearDB();
        buffer = [];
        chunks = [];
        frameCount = 0;
        self.postMessage({ type: 'init-complete' });
        break;

      case 'capture':
        buffer.push(payload);
        frameCount++;

        // Flush chunk when full
        if (buffer.length >= CHUNK_SIZE) {
          await flushChunk();
        }

        // Send progress update (throttled)
        if (frameCount % 30 === 0) {
          self.postMessage({ type: 'progress', count: frameCount });
        }
        break;

      case 'stop':
        // Flush remaining buffer
        if (buffer.length > 0) {
          await flushChunk();
        }

        // Retrieve all snapshots
        const allSnapshots = await getAllSnapshots();

        if (allSnapshots.length === 0) {
          self.postMessage({ type: 'stop-complete', snapshots: [] });
          return;
        }

        // Build recording with metadata
        const duration = allSnapshots[allSnapshots.length - 1].timestamp_ms - allSnapshots[0].timestamp_ms;
        const recording = {
          version: 1,
          meta: {
            recorded_at: new Date().toISOString(),
            total_frames: allSnapshots.length,
            duration_ms: duration
          },
          snapshots: allSnapshots
        };

        // Serialize in chunks (non-blocking)
        const jsonStart = '{"version":1,"meta":' + JSON.stringify(recording.meta) + ',"snapshots":[';
        const parts = [jsonStart];

        for (let i = 0; i < allSnapshots.length; i++) {
          if (i > 0) parts.push(',');
          parts.push(JSON.stringify(allSnapshots[i]));

          // Report progress every 100 frames
          if (i % 100 === 0) {
            const progress = Math.round((i / allSnapshots.length) * 100);
            self.postMessage({ type: 'serialize-progress', progress });
          }
        }

        parts.push(']}');

        self.postMessage({
          type: 'stop-complete',
          parts,
          totalFrames: allSnapshots.length
        });
        break;

      default:
        console.warn('Unknown message type:', type);
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
  }
};
