document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const placeholder = document.getElementById('results-placeholder');
    const loadingState = document.getElementById('loading-state');
    const resultsContent = document.getElementById('results-content');
    const sampleCards = document.querySelectorAll('.sample-card');
    
    // Cropper elements
    const cropperContainer = document.getElementById('cropper-container');
    const cropCanvas = document.getElementById('crop-canvas');
    const ctx = cropCanvas.getContext('2d');
    const zoomSlider = document.getElementById('zoom-slider');
    const btnCancelCrop = document.getElementById('btn-cancel-crop');
    const btnSubmitCrop = document.getElementById('btn-submit-crop');

    // Results elements
    const imgPreview = document.getElementById('prediction-img-preview');
    const classNameEl = document.getElementById('predicted-class-name');
    const confidenceValEl = document.getElementById('prediction-confidence-val');
    const breakdownChart = document.getElementById('breakdown-chart');
    const tipsIconColor = document.getElementById('tips-icon-color');
    const tipsTitle = document.getElementById('tips-title');
    const tipsList = document.getElementById('tips-list');

    // UI Color mapping
    const CLASS_COLORS = {
        cardboard: 'var(--color-cardboard)',
        glass: 'var(--color-glass)',
        metal: 'var(--color-metal)',
        paper: 'var(--color-paper)',
        plastic: 'var(--color-plastic)',
        trash: 'var(--color-trash)'
    };

    // Cropper state
    let currentImage = null;
    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startDragX = 0;
    let startDragY = 0;
    let originalFileName = 'custom_image.jpg';
    let baseScale = 1.0;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle file browser selection
    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Load selected file into the cropper
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select or drop an image file.');
            return;
        }
        originalFileName = file.name;

        const reader = new FileReader();
        reader.onload = function(e) {
            currentImage = new Image();
            currentImage.onload = function() {
                // Reset states
                zoom = 1.0;
                zoomSlider.value = 1.0;
                panX = 0;
                panY = 0;
                
                // Toggle view
                uploadZone.classList.add('hidden');
                cropperContainer.classList.remove('hidden');
                
                // Calculate cover scaling ratio
                const scaleX = cropCanvas.width / currentImage.width;
                const scaleY = cropCanvas.height / currentImage.height;
                baseScale = Math.max(scaleX, scaleY);
                
                drawCanvas();
            };
            currentImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Canvas drawing function
    function drawCanvas() {
        if (!currentImage) return;
        
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
        
        ctx.save();
        // Translate to center to apply zoom relative to center
        ctx.translate(cropCanvas.width / 2, cropCanvas.height / 2);
        ctx.scale(zoom, zoom);
        // Translate back and apply dragging offset
        ctx.translate(-cropCanvas.width / 2 + panX, -cropCanvas.height / 2 + panY);
        
        // Draw base image centered
        const w = currentImage.width * baseScale;
        const h = currentImage.height * baseScale;
        const x = (cropCanvas.width - w) / 2;
        const y = (cropCanvas.height - h) / 2;
        
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        
        ctx.drawImage(currentImage, x, y, w, h);
        ctx.restore();
    }

    // Mouse Panning interactions
    cropCanvas.addEventListener('mousedown', (e) => {
        if (!currentImage) return;
        isDragging = true;
        cropCanvas.style.cursor = 'grabbing';
        startDragX = e.clientX - panX * zoom;
        startDragY = e.clientY - panY * zoom;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        panX = (e.clientX - startDragX) / zoom;
        panY = (e.clientY - startDragY) / zoom;
        drawCanvas();
    });

    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            cropCanvas.style.cursor = 'grab';
        }
    });

    // Touch support (Mobile)
    cropCanvas.addEventListener('touchstart', (e) => {
        if (!currentImage || e.touches.length === 0) return;
        isDragging = true;
        const touch = e.touches[0];
        startDragX = touch.clientX - panX * zoom;
        startDragY = touch.clientY - panY * zoom;
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDragging || e.touches.length === 0) return;
        const touch = e.touches[0];
        panX = (touch.clientX - startDragX) / zoom;
        panY = (touch.clientY - startDragY) / zoom;
        drawCanvas();
    });

    window.addEventListener('touchend', () => {
        isDragging = false;
    });

    // Zoom slider interaction
    zoomSlider.addEventListener('input', () => {
        zoom = parseFloat(zoomSlider.value);
        drawCanvas();
    });

    // Cancel / Reset Cropper UI
    btnCancelCrop.addEventListener('click', () => {
        currentImage = null;
        fileInput.value = '';
        cropperContainer.classList.add('hidden');
        uploadZone.classList.remove('hidden');
    });

    // Export cropped selection and perform prediction
    btnSubmitCrop.addEventListener('click', () => {
        if (!currentImage) return;
        
        showLoading();
        
        // Create an offscreen canvas to isolate the 260x260 crop area (inner frame overlay boundary)
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = 260;
        exportCanvas.height = 260;
        const exportCtx = exportCanvas.getContext('2d');
        
        // Exclude 20px margins from 300x300 canvas to extract the central 260x260 region
        exportCtx.drawImage(cropCanvas, 20, 20, 260, 260, 0, 0, 260, 260);
        
        exportCanvas.toBlob((blob) => {
            if (!blob) {
                showError('Could not process crop selection.');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', blob, originalFileName);
            
            fetch('/classify', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const objectUrl = URL.createObjectURL(blob);
                    renderResults(data, objectUrl);
                    
                    // Reset UI
                    cropperContainer.classList.add('hidden');
                    uploadZone.classList.remove('hidden');
                    fileInput.value = '';
                } else {
                    showError(data.error);
                }
            })
            .catch(err => {
                console.error(err);
                showError('Network error occurred during classification.');
            });
        }, 'image/jpeg', 0.95);
    });

    // Handle dataset sample selection
    sampleCards.forEach(card => {
        card.addEventListener('click', () => {
            const category = card.getAttribute('data-category');
            const filename = card.getAttribute('data-filename');
            handleSampleSelect(category, filename);
        });
    });

    function handleSampleSelect(category, filename) {
        showLoading();

        const samplePath = `Garbage/original_images/${category}/${filename}`;
        
        fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sample_path: samplePath })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const previewUrl = `/samples/${category}/${filename}`;
                renderResults(data, previewUrl);
            } else {
                showError(data.error);
            }
        })
        .catch(err => {
            console.error(err);
            showError('Network error occurred during classification.');
        });
    }

    function showLoading() {
        placeholder.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loadingState.classList.remove('hidden');
    }

    function showError(msg) {
        placeholder.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        alert(`Error: ${msg}`);
    }

    function renderResults(data, previewUrl) {
        loadingState.classList.add('hidden');
        placeholder.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        imgPreview.src = previewUrl;

        const topPredClass = data.prediction;
        const topPredProb = (data.probability * 100).toFixed(0);
        const accentColor = CLASS_COLORS[topPredClass] || 'var(--text-primary)';

        classNameEl.textContent = topPredClass;
        classNameEl.style.color = accentColor;
        confidenceValEl.textContent = `${topPredProb}%`;
        confidenceValEl.style.color = accentColor;

        breakdownChart.innerHTML = '';
        data.all_predictions.forEach(pred => {
            const pClass = pred.class;
            const pPct = (pred.probability * 100).toFixed(0);
            const pColor = CLASS_COLORS[pClass] || 'var(--text-muted)';
            
            const chartRow = document.createElement('div');
            chartRow.className = 'chart-row';
            chartRow.innerHTML = `
                <div class="row-info">
                    <span class="row-class">${pClass}</span>
                    <span class="row-pct">${pPct}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="background-color: ${pColor};"></div>
                </div>
            `;
            breakdownChart.appendChild(chartRow);
            
            setTimeout(() => {
                const fill = chartRow.querySelector('.bar-fill');
                if (fill) fill.style.width = `${pPct}%`;
            }, 50);
        });

        const tips = data.recycling_info;
        tipsTitle.textContent = tips.title || 'Recycling Guidelines';
        tipsIconColor.style.backgroundColor = tips.color || 'var(--color-trash)';
        tipsIconColor.style.boxShadow = `0 0 15px ${tips.color || 'var(--color-trash)'}`;

        tipsList.innerHTML = '';
        if (tips.tips && tips.tips.length > 0) {
            tips.tips.forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip;
                li.style.color = 'var(--text-secondary)';
                tipsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific guidelines available. Please consult local guidelines.';
            tipsList.appendChild(li);
        }
    }
});
