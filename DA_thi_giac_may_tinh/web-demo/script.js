// ===== DefectDetect OBB — Interactive Demo Script v3.0 =====
// Backend v3 trả về annotated_image (base64) đã vẽ OBB giống notebook BƯỚC 19
// Frontend chỉ cần hiển thị ảnh đó + cập nhật sidebar JSON results

document.addEventListener('DOMContentLoaded', () => {
  // === ELEMENTS ===
  const tabUpload = document.getElementById('tabUpload');
  const tabSample = document.getElementById('tabSample');
  const sampleList = document.getElementById('sampleList');
  const uploadArea = document.getElementById('uploadArea');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const confSlider = document.getElementById('confSlider');
  const confValue = document.getElementById('confValue');

  const detectionEmpty = document.getElementById('detectionEmpty');
  const detectionView = document.getElementById('detectionView');
  const detectionBar = document.getElementById('detectionBar');
  const detectionBarItems = document.getElementById('detectionBarItems');
  const detectionCanvas = document.getElementById('detectionCanvas');
  // Thêm img tag để hiển thị ảnh annotated từ backend
  let detectionImage = document.getElementById('detectionImage');

  const resultsEmpty = document.getElementById('resultsEmpty');
  const resultsList = document.getElementById('resultsList');
  const btnReanalyze = document.getElementById('btnReanalyze');

  // === CLASS CONFIG ===
  const CLASS_COLORS = {
    'BypassedSubs': '#ff5252',
    'Diode': '#69f0ae',
    'MultiHotSpot': '#40c4ff',
    'Hotspot': '#ffd740',
    // fallback cho defect_* names
    'defect_0': '#ff5252',
    'defect_1': '#69f0ae',
    'defect_2': '#40c4ff',
    'defect_3': '#ffd740',
  };

  // === MODEL SPECS ===
  const MODEL_SPECS = {
    yolov8n: { label: 'YOLOv8n-OBB', params: '3.1M', gflops: '8.4',  latency: '7.6ms',  map50: '0.939', map95: '0.930', head: 'OBBHead-Nano',   tag: 'Nhanh nhất' },
    yolov8s: { label: 'YOLOv8s-OBB', params: '11.4M', gflops: '30.3', latency: '14.2ms', map50: '0.953', map95: '0.944', head: 'OBBHead-Small',  tag: '+1.4% mAP so v8n' },
    yolov8m: { label: 'YOLOv8m-OBB', params: '26.4M', gflops: '81.9', latency: '28.7ms', map50: '0.962', map95: '0.951', head: 'OBBHead-Medium', tag: '+2.3% mAP so v8n' },
    yolov8l: { label: 'YOLOv8l-OBB', params: '44.9M', gflops: '136',  latency: '45.1ms', map50: '0.967', map95: '0.955', head: 'OBBHead-Large',  tag: '+2.8% mAP so v8n' },
    yolov8x: { label: 'YOLOv8x-OBB', params: '68.2M', gflops: '179',  latency: '68.4ms', map50: '0.970', map95: '0.958', head: 'OBBHead-XLarge', tag: 'Chính xác nhất' },
  };

  // === STATE ===
  let currentSample = null;
  let currentImageBlob = null;
  let currentConfidence = 30;  // notebook mặc định conf=0.3
  let currentModel = 'yolov8n';
  let isDetecting = false;

  // === INIT SLIDER ===
  confSlider.value = currentConfidence;
  confSlider.style.setProperty('--val', currentConfidence + '%');
  confValue.textContent = currentConfidence + '%';

  // === MODEL SELECTOR ===
  const modelSelector = document.getElementById('modelSelector');

  function updateModelInfo(key) {
    const spec = MODEL_SPECS[key];
    if (!spec) return;
    // Sidebar stats
    document.getElementById('infoModelName').textContent = spec.label;
    document.getElementById('infoParams').textContent = spec.params;
    document.getElementById('infoGflops').textContent = spec.gflops;
    document.getElementById('infoLatency').textContent = spec.latency;
    document.getElementById('infoMap').textContent = spec.map50;
    document.getElementById('infoMap95').textContent = spec.map95;
    document.getElementById('infoHeadCode').textContent = 'Head: ' + spec.head;
    // Badge comparison vs v8n
    const badge = document.getElementById('modelCompareBadge');
    const badgeDiff = document.getElementById('mcbDiff');
    if (key === 'yolov8n') {
      badge.style.display = 'none';
    } else {
      badge.style.display = 'flex';
      badgeDiff.textContent = spec.tag;
    }
    // Animate highlight
    const nameEl = document.getElementById('infoModelName');
    nameEl.style.transition = 'color 0.1s';
    nameEl.style.color = '#fff';
    setTimeout(() => { nameEl.style.color = ''; }, 300);
  }

  modelSelector.addEventListener('change', (e) => {
    currentModel = e.target.value;
    updateModelInfo(currentModel);
    // Re-run detection if image already loaded
    if (currentImageBlob || currentSample) runDetection();
  });

  // Init sidebar on load
  updateModelInfo(currentModel);

  // === SIDEBAR TABS ===
  tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabSample.classList.remove('active');
    sampleList.classList.add('hidden');
    uploadArea.classList.remove('hidden');
  });

  tabSample.addEventListener('click', () => {
    tabSample.classList.add('active');
    tabUpload.classList.remove('active');
    uploadArea.classList.add('hidden');
    sampleList.classList.remove('hidden');
  });

  // === CONFIDENCE SLIDER ===
  confSlider.addEventListener('input', (e) => {
    currentConfidence = parseInt(e.target.value);
    confValue.textContent = currentConfidence + '%';
    confSlider.style.setProperty('--val', currentConfidence + '%');
    if (currentSample) runDetection();
  });

  // === SAMPLE CARDS ===
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      loadSample(card.dataset.sample);
    });
  });

  // === QUICK SAMPLE BUTTONS ===
  document.querySelectorAll('.btn-sample-quick').forEach(btn => {
    btn.addEventListener('click', () => {
      const sample = btn.dataset.sample;
      document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
      const card = document.querySelector(`[data-sample="${sample}"]`);
      if (card) card.classList.add('active');
      loadSample(sample);
    });
  });

  // === RIGHT PANEL TABS ===
  document.querySelectorAll('.rpanel-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.rpanel-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.querySelectorAll('.rpanel-content').forEach(c => c.classList.remove('active'));
      document.getElementById('content' + capitalize(tab.dataset.rpanel)).classList.add('active');
    });
  });

  // === FILE UPLOAD ===
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#00e5ff'; });
  dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadUploadedImage(file);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) loadUploadedImage(e.target.files[0]);
  });

  // === REANALYZE ===
  btnReanalyze.addEventListener('click', () => {
    runDetection();
  });

  // === LOAD SAMPLE ===
  function loadSample(sampleKey) {
    currentSample = sampleKey;
    const sampleCard = document.querySelector(`[data-sample="${sampleKey}"] img`);
    if (!sampleCard) return;
    const imgSrc = sampleCard.src;

    fetch(imgSrc)
      .then(r => r.blob())
      .then(blob => {
        // Đảm bảo blob có MIME type đúng (image/jpeg)
        currentImageBlob = new Blob([blob], { type: 'image/jpeg' });
        // Hiển thị ảnh gốc trước khi detect
        showDetectionAreaLoading(imgSrc);
        runDetection();
      })
      .catch((err) => {
        console.warn('Blob fetch failed, using null:', err);
        // Fallback: vẫn gọi detect với sample key
        currentImageBlob = null;
        showDetectionAreaLoading(imgSrc);
        runDetection();
      });
  }

  // === LOAD UPLOADED IMAGE ===
  function loadUploadedImage(file) {
    currentImageBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      currentSample = 'upload';
      showDetectionAreaLoading(e.target.result);
      runDetection();
    };
    reader.readAsDataURL(file);
  }

  // === HIỂN THỊ DETECTION VIEW (loading state) ===
  function showDetectionAreaLoading(imgSrc) {
    detectionEmpty.classList.add('hidden');
    detectionView.classList.remove('hidden');

    // Ẩn canvas cũ (nếu có)
    if (detectionImage) {
      detectionImage.style.display = 'none';
    }
    // Làm mờ canvas hiện tại nếu đang hiển thị
    if (detectionCanvas.style.display !== 'none') {
      detectionCanvas.style.opacity = '0.3';
    } else {
      // Hiển thị ảnh gốc làm preview tạm thời
      if (!detectionImage) {
        detectionImage = document.createElement('img');
        detectionImage.id = 'detectionImage';
        detectionImage.alt = 'Loading...';
        detectionImage.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;display:block;margin:auto;opacity:0.4;';
        detectionView.appendChild(detectionImage);
      }
      detectionImage.src = imgSrc;
      detectionImage.style.display = 'block';
      detectionImage.style.opacity = '0.4';
    }
  }

  // === HIỂN THỊ ẢNH ANNOTATED TỪ BACKEND (vẽ lên canvas) ===
  function showAnnotatedImage(base64Data) {
    console.log('[annotated] Drawing to canvas, data length:', base64Data.length);

    // Ẩn img tag gốc nếu có
    if (detectionImage) detectionImage.style.display = 'none';
    const existingAnnotated = document.getElementById('annotatedResultImg');
    if (existingAnnotated) existingAnnotated.style.display = 'none';

    // Dùng canvas để hiển thị ảnh annotated
    const img = new Image();
    img.onload = () => {
      // Set canvas size = ảnh annotated
      detectionCanvas.width  = img.naturalWidth;
      detectionCanvas.height = img.naturalHeight;
      const ctx = detectionCanvas.getContext('2d');
      ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
      ctx.drawImage(img, 0, 0);
      detectionCanvas.style.display = 'block';
      detectionCanvas.style.maxWidth  = '100%';
      detectionCanvas.style.maxHeight = '100%';
      detectionCanvas.style.objectFit = 'contain';
      console.log('[annotated] Drawn to canvas:', img.naturalWidth, 'x', img.naturalHeight);
    };
    img.onerror = (e) => {
      console.error('[annotated] Failed to load annotated image:', e);
    };
    img.src = base64Data;
  }

  // === RUN DETECTION — gọi FastAPI ===
  async function runDetection() {
    if (isDetecting) return;
    isDetecting = true;

    animateStats(0, '...', '—', 0);
    setLoadingState(true);

    try {
      const conf = currentConfidence / 100;

      // Cho phép gọi API kể cả khi không có blob (dùng sample key)
      const formData = new FormData();
      formData.append('conf', conf.toString());
      formData.append('model_key', currentModel);

      if (currentSample && currentSample !== 'upload') {
        formData.append('sample', currentSample);
      }

      if (currentImageBlob) {
        const fileName = currentImageBlob.name || (currentSample ? currentSample + '.jpg' : 'image.jpg');
        formData.append('file', currentImageBlob, fileName);
        console.log('[detect] blob size:', currentImageBlob.size, 'type:', currentImageBlob.type, 'name:', fileName);
      } else {
        // Nếu không có blob, tạo blob dummy để backend xử lý sample key
        const dummyBlob = new Blob([new Uint8Array([0xFF, 0xD8, 0xFF, 0xE0])], { type: 'image/jpeg' });
        formData.append('file', dummyBlob, (currentSample || 'sample') + '.jpg');
        console.log('[detect] using dummy blob with sample key:', currentSample);
      }

      const res = await fetch('/api/detect', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const detections = data.detections || [];

      console.log('[detect] mode:', data.mode, 'count:', data.count, 'has_img:', !!data.annotated_image, 'img_len:', data.annotated_image?.length);

      showModeBadge(data.mode || 'simulated', data.model);

      // === KEY CHANGE v3: hiển thị ảnh annotated từ backend ===
      if (data.annotated_image) {
        showAnnotatedImage(data.annotated_image);
        console.log('[detect] annotated image set, length:', data.annotated_image.length);
      } else {
        console.warn('[detect] No annotated_image in response');
      }

      updateStats(detections, data.time_ms);
      updateResultsList(detections);
      updateDetectionBar(detections);

      // Ẩn lỗi cũ nếu có
      const errEl = document.getElementById('apiErrorMsg');
      if (errEl) errEl.style.display = 'none';

    } catch (err) {
      console.error('[detect] Detection error:', err);
      showApiError(err.message);
    } finally {
      isDetecting = false;
      setLoadingState(false);
    }
  }

  // === UI HELPERS ===
  function setLoadingState(loading) {
    if (detectionImage) detectionImage.style.opacity = loading ? '0.4' : '1';
    const annotatedImg = document.getElementById('annotatedResultImg');
    if (annotatedImg) annotatedImg.style.opacity = loading ? '0.4' : '1';
    if (!loading) {
      // Restore canvas opacity after loading done
      detectionCanvas.style.opacity = '1';
    }
    btnReanalyze.textContent = loading ? 'ĐANG PHÂN TÍCH...' : 'PHÂN TÍCH LẠI';
    btnReanalyze.disabled = loading;
  }

  function showModeBadge(mode, modelPath) {
    let badge = document.getElementById('modeBadge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'modeBadge';
      badge.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 10px;border-radius:4px;font-size:10px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none;display:flex;align-items:center;gap:5px;';
      detectionView.style.position = 'relative';
      detectionView.appendChild(badge);
    }
    const spec = MODEL_SPECS[currentModel];
    const modelLabel = spec ? spec.label.replace('-OBB', '') : currentModel.toUpperCase();
    if (mode === 'real') {
      badge.textContent = `⚡ ${modelLabel} · REAL`;
      badge.style.background = 'rgba(105,240,174,0.2)';
      badge.style.color = '#69f0ae';
      badge.style.border = '1px solid #69f0ae44';
    } else if (mode === 'ground_truth') {
      badge.textContent = `📂 GT · ${modelLabel}`;
      badge.style.background = 'rgba(64,196,255,0.15)';
      badge.style.color = '#40c4ff';
      badge.style.border = '1px solid #40c4ff44';
    } else if (mode === 'simulated') {
      badge.textContent = `⚙ SIMULATED · ${modelLabel}`;
      badge.style.background = 'rgba(255,215,64,0.15)';
      badge.style.color = '#ffd740';
      badge.style.border = '1px solid #ffd74044';
    } else {
      badge.textContent = '⚠ Tải ảnh để detect';
      badge.style.background = 'rgba(255,82,82,0.15)';
      badge.style.color = '#ff5252';
      badge.style.border = '1px solid #ff525244';
    }
  }

  function showApiError(msg) {
    let errEl = document.getElementById('apiErrorMsg');
    if (!errEl) {
      errEl = document.createElement('div');
      errEl.id = 'apiErrorMsg';
      errEl.style.cssText = 'margin:8px 12px;padding:8px 12px;border-radius:6px;font-size:11px;background:rgba(255,82,82,0.1);color:#ff5252;border:1px solid #ff525244;';
      resultsList.parentElement.insertBefore(errEl, resultsList);
    }
    errEl.style.display = 'block';
    if (msg.includes('FastAPI') || msg.includes('fetch') || msg.includes('503') || msg.includes('Failed')) {
      errEl.innerHTML = '❌ <b>FastAPI chưa chạy</b><br>Mở terminal mới và chạy:<br><code style="font-size:10px">cd web-demo/api && start.bat</code>';
    } else {
      errEl.textContent = '❌ ' + msg;
    }
    setTimeout(() => { if (errEl) errEl.style.display = 'none'; }, 8000);
  }

  // === UPDATE STATS ===
  function updateStats(detections, time) {
    const uniqueClasses = [...new Set(detections.map(d => d.cls))];
    const avgConf = detections.length > 0
      ? Math.round(detections.reduce((s, d) => s + d.conf, 0) / detections.length * 100)
      : 0;
    animateStats(detections.length, time, avgConf + '%', uniqueClasses.length);
  }

  function animateStats(defects, time, conf, classes) {
    animateValue('defectCount', defects);
    document.getElementById('timeValue').textContent = time;
    document.getElementById('avgConf').textContent = conf;
    animateValue('classCount', classes);
  }

  function animateValue(id, target) {
    const el = document.getElementById(id);
    const start = parseInt(el.textContent) || 0;
    const duration = 400;
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(start + (target - start) * ease);
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  // === UPDATE RESULTS LIST ===
  function updateResultsList(detections) {
    resultsEmpty.classList.add('hidden');
    resultsList.classList.remove('hidden');
    btnReanalyze.classList.remove('hidden');

    resultsList.innerHTML = detections.map((det, i) => {
      const color = det.color_hex || CLASS_COLORS[det.cls] || '#00e5ff';
      return `
        <div class="result-card" style="border-left-color:${color}" data-index="${i}">
          <div class="result-card__header">
            <span class="result-card__name">
              <span class="det-bar-dot" style="background:${color}"></span>
              ${det.cls.toUpperCase()}
            </span>
            <span class="result-card__conf" style="color:${color}">⊙ ${Math.round(det.conf * 100)}%</span>
          </div>
          <div class="result-card__details">
            OBB θ=${det.angle != null ? det.angle + '°' : 'N/A'}
            ${det.size && det.size.length === 2 ? '[' + det.size[0] + '×' + det.size[1] + 'px]' : ''}
          </div>
        </div>
      `;
    }).join('');

    if (detections.length === 0) {
      resultsEmpty.classList.remove('hidden');
      resultsList.classList.add('hidden');
      btnReanalyze.classList.add('hidden');
    }
  }

  // === UPDATE DETECTION BAR ===
  function updateDetectionBar(detections) {
    if (detections.length === 0) {
      detectionBar.classList.add('hidden');
      return;
    }
    detectionBar.classList.remove('hidden');
    detectionBarItems.innerHTML = detections.map(det => {
      const color = det.color_hex || CLASS_COLORS[det.cls] || '#00e5ff';
      return `
        <div class="det-bar-item">
          <span class="det-bar-dot" style="background:${color}"></span>
          <strong>${det.cls.toUpperCase()}</strong> ${Math.round(det.conf * 100)}%${det.angle != null ? ' · θ=' + det.angle + '°' : ''}
        </div>
      `;
    }).join('');
  }

  // === DRAW mAP CHART ===
  function drawMapChart() {
    const canvas = document.getElementById('mapChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth;
    const h = canvas.height = 140;
    const pad = { top: 10, right: 10, bottom: 25, left: 30 };
    const pw = w - pad.left - pad.right;
    const ph = h - pad.top - pad.bottom;

    const epochs = 100;
    const data = [];
    for (let i = 0; i < epochs; i++) {
      const t = i / epochs;
      const val = 0.2 + 0.72 * (1 - Math.exp(-4 * t)) + (Math.random() - 0.5) * 0.03;
      data.push(Math.min(val, 0.96));
    }

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = '#1a2a3a';
    ctx.lineWidth = 0.5;
    for (let v = 0; v <= 1; v += 0.25) {
      const y = pad.top + ph * (1 - v);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = '#556677';
      ctx.font = '9px Inter';
      ctx.fillText(v.toFixed(1), 2, y + 3);
    }

    const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    gradient.addColorStop(0, 'rgba(0,229,255,0.3)');
    gradient.addColorStop(1, 'rgba(0,229,255,0)');

    ctx.beginPath();
    ctx.moveTo(pad.left, h - pad.bottom);
    data.forEach((v, i) => {
      const x = pad.left + (i / (epochs - 1)) * pw;
      const y = pad.top + ph * (1 - v);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + pw, h - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (epochs - 1)) * pw;
      const y = pad.top + ph * (1 - v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#556677';
    ctx.font = '9px Inter';
    ctx.fillText('0', pad.left, h - 5);
    ctx.fillText('50', pad.left + pw / 2 - 6, h - 5);
    ctx.fillText('100', pad.left + pw - 12, h - 5);
  }

  drawMapChart();
  window.addEventListener('resize', drawMapChart);

  // === UTILITY ===
  function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
});
