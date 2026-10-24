/**
 * TwoHearts High-Performance 60 FPS HTML5 Canvas ECG Waveform Renderer.
 * Supports stacked multi-lead display, baseline scaling, and Grad-CAM saliency heatmaps.
 */

class ECGCanvasViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.data = null;
    this.xaiMode = 'none'; // 'none' | 'fetal' | 'maternal'
    this.xaiOpacity = 0.5;
    this.selectedLead = 'all'; // 'all' | 0 | 1 | 2 | 3

    this.leadColors = [
      '#10B981', // Lead 1: Emerald
      '#06B6D4', // Lead 2: Cyan
      '#3B82F6', // Lead 3: Blue
      '#A855F7'  // Lead 4: Purple
    ];

    this.leadNames = ['Abdomen 1', 'Abdomen 2', 'Abdomen 3', 'Abdomen 4'];

    this.setupResize();
  }

  setupResize() {
    const resize = () => {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.ctx.scale(dpr, dpr);
      this.render();
    };
    window.addEventListener('resize', resize);
    setTimeout(resize, 50);
  }

  setData(waveformData) {
    this.data = waveformData;
    this.render();
  }

  setXAIMode(mode) {
    this.xaiMode = mode;
    this.render();
  }

  setXAIOpacity(opacity) {
    this.xaiOpacity = opacity;
    this.render();
  }

  render() {
    const width = this.canvas.getBoundingClientRect().width;
    const height = this.canvas.getBoundingClientRect().height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    if (!this.data || !this.data.time || this.data.time.length === 0) {
      ctx.fillStyle = '#6B7280';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Select a patient recording to load ECG waveforms...', width / 2, height / 2);
      return;
    }

    const leads = [
      this.data.lead_1,
      this.data.lead_2,
      this.data.lead_3,
      this.data.lead_4
    ];

    const time = this.data.time;
    const nPoints = time.length;
    const numLeads = leads.length;
    const leadHeight = height / numLeads;

    // 1. Draw Grad-CAM Heatmap Background (if enabled)
    if (this.xaiMode !== 'none') {
      const cam = this.xaiMode === 'fetal' ? this.data.cam_fetal : this.data.cam_maternal;
      if (cam && cam.length > 0) {
        this.renderGradCAMHeatmap(ctx, cam, width, height);
      }
    }

    // 2. Draw each abdominal lead waveform
    leads.forEach((sig, leadIdx) => {
      const centerY = leadHeight * leadIdx + leadHeight / 2;
      const color = this.leadColors[leadIdx];

      // Draw zero baseline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Lead Label
      ctx.fillStyle = color;
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(this.leadNames[leadIdx], 12, centerY - leadHeight * 0.32);

      // Scale factor: standardize amplitude to fit lead slot comfortably
      const yAmpScale = leadHeight * 0.18; // z-score ~3-4 fits within height

      // Draw ECG trace
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = 'round';
      ctx.beginPath();

      for (let i = 0; i < nPoints; i++) {
        const x = (i / (nPoints - 1)) * width;
        const val = sig[i];
        const y = centerY - val * yAmpScale;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });

    // 3. Draw Time Scale at the bottom
    this.renderTimeScale(ctx, time[0], time[time.length - 1], width, height);
  }

  renderGradCAMHeatmap(ctx, cam, width, height) {
    const n = cam.length;
    const stepPx = Math.max(2, Math.floor(width / n));

    for (let i = 0; i < n; i++) {
      const saliency = cam[i];
      if (saliency > 0.15) { // Only highlight significant regions
        const x = (i / (n - 1)) * width;
        
        // Color interpolation: Low saliency = yellow-orange, High saliency = bright crimson
        const alpha = Math.min(1.0, saliency * this.xaiOpacity * 0.85);
        
        let r = 239, g = 68, b = 68; // Crimson
        if (saliency < 0.6) {
          // Amber / Orange
          r = 245; g = 158; b = 11;
        }

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.fillRect(x, 0, stepPx + 1, height);
      }
    }
  }

  renderTimeScale(ctx, startTime, endTime, width, height) {
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';

    const numTicks = 10;
    for (let i = 0; i <= numTicks; i++) {
      const x = (i / numTicks) * width;
      const t = (startTime + (i / numTicks) * (endTime - startTime)).toFixed(1);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, height - 6);
      ctx.lineTo(x, height);
      ctx.stroke();

      if (i > 0 && i < numTicks) {
        ctx.fillText(`${t}s`, x, height - 10);
      }
    }
  }
}
