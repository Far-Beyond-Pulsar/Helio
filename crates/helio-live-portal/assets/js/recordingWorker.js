// recordingWorker.js - Extreme performance in-memory recording
'use strict';

// Pure in-memory storage (no IndexedDB during recording!)
let snapshots = [];
let frameCount = 0;

// Message handler
self.onmessage = (e) => {
  const { type, payload } = e.data;

  try {
    switch (type) {
      case 'init':
        // Reset state
        snapshots = [];
        frameCount = 0;
        self.postMessage({ type: 'init-complete' });
        break;

      case 'capture-batch':
        // Receive batched snapshots for minimal message overhead
        if (Array.isArray(payload)) {
          snapshots.push(...payload);
          frameCount += payload.length;

          // Send progress update
          self.postMessage({ type: 'progress', count: frameCount });
        }
        break;

      case 'stop':
        if (snapshots.length === 0) {
          self.postMessage({ type: 'stop-complete', parts: [] });
          return;
        }

        // Serialize in chunks with yield points
        const first = snapshots[0];
        const last = snapshots[snapshots.length - 1];
        const duration = last.timestamp_ms - first.timestamp_ms;

        const meta = {
          recorded_at: new Date().toISOString(),
          total_frames: snapshots.length,
          duration_ms: duration
        };

        // Stream serialization
        const jsonStart = '{"version":1,"meta":' + JSON.stringify(meta) + ',"snapshots":[';
        const parts = [jsonStart];

        // Serialize in batches
        const BATCH_SIZE = 50;
        for (let i = 0; i < snapshots.length; i += BATCH_SIZE) {
          const end = Math.min(i + BATCH_SIZE, snapshots.length);

          for (let j = i; j < end; j++) {
            if (j > 0) parts.push(',');
            parts.push(JSON.stringify(snapshots[j]));
          }

          // Report progress
          const progress = Math.round((end / snapshots.length) * 100);
          self.postMessage({ type: 'serialize-progress', progress });
        }

        parts.push(']}');

        self.postMessage({
          type: 'stop-complete',
          parts,
          totalFrames: snapshots.length
        });

        // Clear memory
        snapshots = [];
        frameCount = 0;
        break;

      default:
        console.warn('Unknown message type:', type);
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message });
  }
};
