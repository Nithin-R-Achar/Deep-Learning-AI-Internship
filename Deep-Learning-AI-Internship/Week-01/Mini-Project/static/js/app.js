document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d');
    const emptyOverlay = document.getElementById('emptyCanvasOverlay');
    const clearBtn = document.getElementById('clearBtn');
    const undoBtn = document.getElementById('undoBtn');
    const strokeWidthInput = document.getElementById('strokeWidth');
    const strokeValueLabel = document.getElementById('strokeValue');
    const presetContainer = document.getElementById('presetButtons');
    
    // Result Elements
    const predictedDigitEl = document.getElementById('predictedDigit');
    const confidenceValueEl = document.getElementById('confidenceValue');
    const confidenceMeterBar = document.getElementById('confidenceMeterBar');
    const predictionCaptionEl = document.getElementById('predictionCaption');
    const latencyBadgeEl = document.getElementById('latencyBadge');
    const probabilityBarsContainer = document.getElementById('probabilityBars');
    const processedPreview = document.getElementById('processedPreview');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const modelStatusBadge = document.getElementById('modelStatusBadge');
    const modelStatusText = document.getElementById('modelStatusText');
    const footerAccuracy = document.getElementById('footerAccuracy');

    // Drawing State
    let isDrawing = false;
    let strokeWidth = parseInt(strokeWidthInput.value, 10);
    let history = [];
    const maxHistory = 20;
    let predictTimeout = null;
    let hasStrokes = false;

    // Initialize Canvas
    function initCanvas() {
        ctx.fillStyle = '#030712'; // Black background matching MNIST background
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#FFFFFF'; // White stroke matching MNIST digits
        ctx.lineWidth = strokeWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        saveState();
    }

    // Save history state
    function saveState() {
        if (history.length >= maxHistory) {
            history.shift();
        }
        history.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }

    // Initialize Probability Bars (0 to 9)
    function initProbabilityBars() {
        probabilityBarsContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.id = `prob-row-${i}`;
            row.innerHTML = `
                <span class="prob-digit">${i}</span>
                <div class="prob-track">
                    <div class="prob-fill" id="prob-fill-${i}"></div>
                </div>
                <span class="prob-value" id="prob-val-${i}">0.0%</span>
            `;
            probabilityBarsContainer.appendChild(row);
        }
    }

    // Quick Preset Digits (vectors)
    const presetDigits = {
        0: [[140, 60], [80, 110], [80, 170], [140, 220], [200, 170], [200, 110], [140, 60]],
        1: [[110, 80], [140, 60], [140, 220], [110, 220], [170, 220]],
        2: [[90, 90], [140, 60], [190, 90], [190, 120], [85, 215], [195, 215]],
        3: [[90, 70], [185, 70], [135, 130], [185, 160], [160, 215], [95, 205]],
        4: [[170, 220], [170, 60], [80, 160], [200, 160]],
        5: [[180, 70], [105, 70], [100, 130], [180, 140], [180, 205], [95, 215]],
        6: [[175, 75], [115, 125], [95, 175], [145, 220], [185, 185], [145, 145], [95, 175]],
        7: [[85, 70], [195, 70], [125, 220]],
        8: [[140, 135], [100, 95], [140, 60], [180, 95], [140, 135], [95, 180], [140, 220], [185, 180], [140, 135]],
        9: [[185, 110], [140, 60], [95, 105], [135, 145], [185, 110], [155, 215], [125, 220]]
    };

    function initPresets() {
        presetContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const btn = document.createElement('button');
            btn.className = 'preset-btn';
            btn.textContent = i;
            btn.title = `Draw preset digit ${i}`;
            btn.addEventListener('click', () => drawPreset(i));
            presetContainer.appendChild(btn);
        }
    }

    function drawPreset(digit) {
        clearCanvas();
        const points = presetDigits[digit];
        if (!points || points.length === 0) return;

        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }
        ctx.stroke();
        hasStrokes = true;
        emptyOverlay.classList.add('hidden');
        saveState();
        triggerPrediction();
    }

    // Coordinate calculation
    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    // Drawing Handlers
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasStrokes = true;
        emptyOverlay.classList.add('hidden');
        
        const coords = getCanvasCoords(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        ctx.lineTo(coords.x, coords.y); // Draw point if just clicked
        ctx.stroke();

        schedulePrediction(180);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const coords = getCanvasCoords(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        schedulePrediction(160);
    }

    function stopDrawing(e) {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
        saveState();
        triggerPrediction();
    }

    // Canvas Events (Mouse & Touch)
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);

    // Controls
    function clearCanvas() {
        ctx.fillStyle = '#030712';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasStrokes = false;
        emptyOverlay.classList.remove('hidden');
        history = [];
        saveState();
        resetPredictionUI();
    }

    function undoStroke() {
        if (history.length > 1) {
            history.pop(); // Remove current
            const prevState = history[history.length - 1];
            ctx.putImageData(prevState, 0, 0);
            hasStrokes = true;
            emptyOverlay.classList.add('hidden');
            triggerPrediction();
        } else if (history.length === 1) {
            clearCanvas();
        }
    }

    clearBtn.addEventListener('click', clearCanvas);
    undoBtn.addEventListener('click', undoStroke);

    // Stroke width slider
    strokeWidthInput.addEventListener('input', (e) => {
        strokeWidth = parseInt(e.target.value, 10);
        strokeValueLabel.textContent = `${strokeWidth}px`;
        ctx.lineWidth = strokeWidth;
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'c' || e.key === 'C') {
            clearCanvas();
        } else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
            e.preventDefault();
            undoStroke();
        }
    });

    // Prediction Logic
    function schedulePrediction(delay = 150) {
        if (predictTimeout) clearTimeout(predictTimeout);
        predictTimeout = setTimeout(triggerPrediction, delay);
    }

    async function triggerPrediction() {
        if (predictTimeout) clearTimeout(predictTimeout);

        const dataUrl = canvas.toDataURL('image/png');
        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });

            const latency = Math.round(performance.now() - startTime);
            latencyBadgeEl.textContent = `Latency: ${latency} ms`;

            if (!response.ok) {
                const errData = await response.json();
                console.warn('Prediction warning:', errData.message);
                return;
            }

            const data = await response.json();

            if (data.status === 'empty') {
                resetPredictionUI();
            } else if (data.status === 'success') {
                updatePredictionUI(data);
            }
        } catch (err) {
            console.error('Inference error:', err);
        }
    }

    function updatePredictionUI(data) {
        const digit = data.prediction;
        const confidence = data.confidence;
        const probs = data.probabilities;

        // Top prediction
        predictedDigitEl.textContent = digit;
        confidenceValueEl.textContent = `${confidence.toFixed(1)}%`;
        confidenceMeterBar.style.width = `${confidence}%`;
        predictionCaptionEl.textContent = confidence > 85 
            ? 'High confidence prediction' 
            : confidence > 55 
            ? 'Moderate confidence prediction' 
            : 'Low confidence / ambiguous stroke';

        // Probability bars
        for (let i = 0; i < 10; i++) {
            const row = document.getElementById(`prob-row-${i}`);
            const fill = document.getElementById(`prob-fill-${i}`);
            const val = document.getElementById(`prob-val-${i}`);
            const p = probs[i];

            fill.style.width = `${p}%`;
            val.textContent = `${p.toFixed(1)}%`;

            if (i === digit) {
                row.classList.add('winner');
            } else {
                row.classList.remove('winner');
            }
        }

        // 28x28 Preview
        if (data.preview_image) {
            processedPreview.src = data.preview_image;
            processedPreview.classList.remove('hidden');
            previewPlaceholder.style.display = 'none';
        }
    }

    function resetPredictionUI() {
        predictedDigitEl.textContent = '—';
        confidenceValueEl.textContent = '0.0%';
        confidenceMeterBar.style.width = '0%';
        predictionCaptionEl.textContent = 'Draw a number to predict';
        latencyBadgeEl.textContent = 'Latency: -- ms';

        for (let i = 0; i < 10; i++) {
            const row = document.getElementById(`prob-row-${i}`);
            const fill = document.getElementById(`prob-fill-${i}`);
            const val = document.getElementById(`prob-val-${i}`);
            if (row) row.classList.remove('winner');
            if (fill) fill.style.width = '0%';
            if (val) val.textContent = '0.0%';
        }

        processedPreview.classList.add('hidden');
        previewPlaceholder.style.display = 'flex';
    }

    // Check Model Metadata
    async function checkModelMeta() {
        try {
            const res = await fetch('/meta');
            const data = await res.json();
            if (data.status === 'success' && data.data) {
                const meta = data.data;
                modelStatusBadge.classList.remove('pending');
                modelStatusText.textContent = `Model Ready (${meta.best_accuracy}% Test Acc)`;
                footerAccuracy.innerHTML = `Test Accuracy: <strong>${meta.best_accuracy}%</strong>`;
            } else {
                modelStatusText.textContent = 'Training / Initializing...';
                setTimeout(checkModelMeta, 3000);
            }
        } catch (e) {
            modelStatusText.textContent = 'Connecting...';
            setTimeout(checkModelMeta, 4000);
        }
    }

    // Start
    initCanvas();
    initProbabilityBars();
    initPresets();
    checkModelMeta();
});
