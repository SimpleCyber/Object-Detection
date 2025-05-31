class GeminiCameraApp {
    constructor() {
        this.currentStream = null;
        this.currentCamera = 'user'; // 'user' for front, 'environment' for back
        this.isFlashOn = false;
        this.timerSeconds = 0;
        this.timerInterval = null;
        this.currentAnalysis = null;
        this.speechSynthesis = window.speechSynthesis;
        this.voiceSettings = {
            speed: 1,
            language: 'en-US'
        };
        
        this.init();
    }

    async init() {
        await this.initCamera();
        this.loadSettings();
        this.loadGallery();
        this.loadHistory();
        this.setupEventListeners();
    }

    async initCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: this.currentCamera,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            };

            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            const video = document.getElementById('cameraFeed');
            video.srcObject = this.currentStream;
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError('Camera access denied. Please allow camera permissions.');
        }
    }

    setupEventListeners() {
        // Touch focus
        const video = document.getElementById('cameraFeed');
        video.addEventListener('click', (e) => this.handleFocus(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                this.capturePhoto();
            }
        });

        // Settings
        document.getElementById('voiceSpeed').addEventListener('change', (e) => {
            this.voiceSettings.speed = parseFloat(e.target.value);
            this.saveSettings();
        });

        document.getElementById('voiceLanguage').addEventListener('change', (e) => {
            this.voiceSettings.language = e.target.value;
            this.saveSettings();
        });
    }

    handleFocus(e) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const focusIndicator = document.getElementById('focusIndicator');
        focusIndicator.style.left = (x - 40) + 'px';
        focusIndicator.style.top = (y - 40) + 'px';
        focusIndicator.classList.add('active');
        
        setTimeout(() => {
            focusIndicator.classList.remove('active');
        }, 1000);
    }

    async flipCamera() {
        this.currentCamera = this.currentCamera === 'user' ? 'environment' : 'user';
        
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
        }
        
        await this.initCamera();
        
        // Add flip animation
        const video = document.getElementById('cameraFeed');
        video.style.transform = 'scaleX(-1)';
        setTimeout(() => {
            video.style.transform = 'scaleX(1)';
        }, 300);
    }

    toggleFlash() {
        this.isFlashOn = !this.isFlashOn;
        const flashIcon = document.getElementById('flashIcon');
        const flashBtn = flashIcon.parentElement;
        
        if (this.isFlashOn) {
            flashBtn.classList.add('active');
            flashIcon.className = 'fas fa-bolt';
        } else {
            flashBtn.classList.remove('active');
            flashIcon.className = 'fas fa-bolt';
        }
    }

    toggleTimer() {
        const timerIcon = document.getElementById('timerIcon');
        const timerBtn = timerIcon.parentElement;
        
        if (this.timerSeconds === 0) {
            this.timerSeconds = 3;
            timerBtn.classList.add('active');
            timerIcon.className = 'fas fa-clock';
        } else if (this.timerSeconds === 3) {
            this.timerSeconds = 10;
            timerIcon.className = 'fas fa-clock';
        } else {
            this.timerSeconds = 0;
            timerBtn.classList.remove('active');
            timerIcon.className = 'fas fa-clock';
        }
    }

    async capturePhoto() {
        const captureBtn = document.getElementById('captureBtn');
        captureBtn.classList.add('pulse');
        
        if (this.timerSeconds > 0) {
            await this.startTimer();
        }
        
        const video = document.getElementById('cameraFeed');
        const canvas = document.getElementById('photoCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Flash effect
        if (this.isFlashOn) {
            document.body.style.background = '#fff';
            setTimeout(() => {
                document.body.style.background = '#000';
            }, 100);
        }
        
        ctx.drawImage(video, 0, 0);
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
        
        // Save to gallery
        this.saveToGallery(imageDataUrl);
        
        // Analyze with Gemini
        await this.analyzeImage(imageDataUrl);
        
        setTimeout(() => {
            captureBtn.classList.remove('pulse');
        }, 300);
    }

    async startTimer() {
        return new Promise((resolve) => {
            let countdown = this.timerSeconds;
            const timerDisplay = document.createElement('div');
            timerDisplay.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 4rem;
                font-weight: bold;
                color: white;
                z-index: 1000;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            `;
            document.body.appendChild(timerDisplay);
            
            const interval = setInterval(() => {
                timerDisplay.textContent = countdown;
                countdown--;
                
                if (countdown < 0) {
                    clearInterval(interval);
                    document.body.removeChild(timerDisplay);
                    resolve();
                }
            }, 1000);
        });
    }

    async analyzeImage(imageDataUrl) {
        const analysisPanel = document.getElementById('analysisPanel');
        const analysisLoading = document.getElementById('analysisLoading');
        const analysisResults = document.getElementById('analysisResults');
        
        analysisPanel.classList.add('active');
        analysisLoading.style.display = 'block';
        analysisResults.classList.remove('active');
        
        try {
            // Convert data URL to base64
            const base64Image = imageDataUrl.split(',')[1];
            
            const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${this.getApiKey()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [
                            {
                                text: "Analyze this image and provide exactly 2 key insights about the objects, people, or scene. Format your response as two separate points, each focusing on different aspects like object identification, scene description, colors, emotions, or interesting details. Keep each point concise but informative."
                            },
                            {
                                inline_data: {
                                    mime_type: "image/jpeg",
                                    data: base64Image
                                }
                            }
                        ]
                    }]
                })
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }

            const data = await response.json();
            const analysisText = data.candidates[0].content.parts[0].text;
            
            this.currentAnalysis = analysisText;
            this.displayAnalysis(analysisText);
            this.saveToHistory(imageDataUrl, analysisText);
            
        } catch (error) {
            console.error('Error analyzing image:', error);
            this.showError('Failed to analyze image. Please check your internet connection.');
        }
        
        analysisLoading.style.display = 'none';
    }

    getApiKey() {
        // In a real app, this would be securely stored
        return 'AIzaSyA36qY9SbEzYsZsCRMiCSKIcnr5PRkkpgM';
    }

    displayAnalysis(analysisText) {
        const analysisResults = document.getElementById('analysisResults');
        
        // Split the analysis into points
        const points = analysisText.split(/\d+\.|\n/).filter(point => point.trim().length > 0);
        const limitedPoints = points.slice(0, 2); // Limit to 2 points
        
        let html = '';
        limitedPoints.forEach((point, index) => {
            html += `
                <div class="analysis-item">
                    <h4><i class="fas fa-eye"></i> Insight ${index + 1}</h4>
                    <p>${point.trim()}</p>
                </div>
            `;
        });
        
        analysisResults.innerHTML = html;
        analysisResults.classList.add('active');
    }

    speakAnalysis() {
        if (!this.currentAnalysis) return;
        
        // Stop any ongoing speech
        this.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(this.currentAnalysis);
        utterance.rate = this.voiceSettings.speed;
        utterance.lang = this.voiceSettings.language;
        
        // Find the best voice for the selected language
        const voices = this.speechSynthesis.getVoices();
        const voice = voices.find(v => v.lang.startsWith(this.voiceSettings.language.split('-')[0]));
        if (voice) {
            utterance.voice = voice;
        }
        
        this.speechSynthesis.speak(utterance);
        
        // Visual feedback
        const speakBtn = document.querySelector('.speak-btn');
        speakBtn.innerHTML = '<i class="fas fa-volume-up fa-pulse"></i> Speaking...';
        speakBtn.disabled = true;
        
        utterance.onend = () => {
            speakBtn.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
            speakBtn.disabled = false;
        };
    }

    closeAnalysis() {
        const analysisPanel = document.getElementById('analysisPanel');
        analysisPanel.classList.remove('active');
        this.speechSynthesis.cancel();
    }

    saveToGallery(imageDataUrl) {
        const gallery = JSON.parse(localStorage.getItem('cameraGallery') || '[]');
        const timestamp = new Date().toISOString();
        
        gallery.unshift({
            id: Date.now(),
            image: imageDataUrl,
            timestamp: timestamp
        });
        
        // Keep only last 50 images
        if (gallery.length > 50) {
            gallery.splice(50);
        }
        
        localStorage.setItem('cameraGallery', JSON.stringify(gallery));
    }

    saveToHistory(imageDataUrl, analysis) {
        const history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
        const timestamp = new Date().toISOString();
        
        history.unshift({
            id: Date.now(),
            image: imageDataUrl,
            analysis: analysis,
            timestamp: timestamp
        });
        
        // Keep only last 100 analyses
        if (history.length > 100) {
            history.splice(100);
        }
        
        localStorage.setItem('analysisHistory', JSON.stringify(history));
    }

    loadGallery() {
        const gallery = JSON.parse(localStorage.getItem('cameraGallery') || '[]');
        const galleryGrid = document.getElementById('galleryGrid');
        
        if (gallery.length === 0) {
            galleryGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-camera"></i>
                    <h3>No Photos Yet</h3>
                    <p>Take your first photo to see it here</p>
                </div>
            `;
            return;
        }
        
        galleryGrid.innerHTML = gallery.map(item => `
            <div class="gallery-item" onclick="app.viewImage('${item.image}')">
                <img src="${item.image}" alt="Photo">
                <div class="gallery-item-overlay">
                    ${new Date(item.timestamp).toLocaleDateString()}
                </div>
            </div>
        `).join('');
    }

    loadHistory() {
        const history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
        const historyList = document.getElementById('historyList');
        
        if (history.length === 0) {
            historyList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-brain"></i>
                    <h3>No Analysis History</h3>
                    <p>Analyze your first photo to see results here</p>
                </div>
            `;
            return;
        }
        
        historyList.innerHTML = history.map(item => `
            <div class="history-item" onclick="app.viewHistoryItem('${item.id}')">
                <div class="history-item-header">
                    <strong><i class="fas fa-brain"></i> Analysis</strong>
                    <span class="history-item-date">${new Date(item.timestamp).toLocaleString()}</span>
                </div>
                <div class="history-item-preview">
                    ${item.analysis.substring(0, 100)}...
                </div>
            </div>
        `).join('');
    }

    viewImage(imageSrc) {
        // Create image viewer modal
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = `
            <div class="modal-content" style="background: rgba(0,0,0,0.9); max-width: 90vw;">
                <div class="modal-header">
                    <h2><i class="fas fa-image"></i> Photo</h2>
                    <button class="close-modal" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div style="padding: 20px; text-align: center;">
                    <img src="${imageSrc}" style="max-width: 100%; max-height: 60vh; border-radius: 12px;">
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    viewHistoryItem(itemId) {
        const history = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
        const item = history.find(h => h.id == itemId);
        
        if (!item) return;
        
        this.currentAnalysis = item.analysis;
        this.displayAnalysis(item.analysis);
        
        const analysisPanel = document.getElementById('analysisPanel');
        analysisPanel.classList.add('active');
        
        this.closeHistory();
    }

    loadSettings() {
        const settings = JSON.parse(localStorage.getItem('appSettings') || '{}');
        this.voiceSettings = { ...this.voiceSettings, ...settings };
        
        document.getElementById('voiceSpeed').value = this.voiceSettings.speed;
        document.getElementById('voiceLanguage').value = this.voiceSettings.language;
    }

    saveSettings() {
        localStorage.setItem('appSettings', JSON.stringify(this.voiceSettings));
    }

    clearAllData() {
        if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
            localStorage.removeItem('cameraGallery');
            localStorage.removeItem('analysisHistory');
            localStorage.removeItem('appSettings');
            
            this.loadGallery();
            this.loadHistory();
            this.loadSettings();
            
            alert('All data cleared successfully!');
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 25px;
            z-index: 2000;
            font-weight: 600;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        `;
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            document.body.removeChild(errorDiv);
        }, 5000);
    }
}

// Global functions for HTML onclick handlers
function openGallery() {
    document.getElementById('galleryModal').classList.add('active');
    app.loadGallery();
}

function closeGallery() {
    document.getElementById('galleryModal').classList.remove('active');
}

function openHistory() {
    document.getElementById('historyModal').classList.add('active');
    app.loadHistory();
}

function closeHistory() {
    document.getElementById('historyModal').classList.remove('active');
}

function openSettings() {
    document.getElementById('settingsModal').classList.add('active');
}

function closeSettings() {
    document.getElementById('settingsModal').classList.remove('active');
}

function switchPage(page) {
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.nav-btn').classList.add('active');
    
    // Handle page switching
    switch(page) {
        case 'camera':
            // Already on camera page
            break;
        case 'gallery':
            openGallery();
            break;
        case 'history':
            openHistory();
            break;
    }
}

function capturePhoto() {
    app.capturePhoto();
}

function flipCamera() {
    app.flipCamera();
}

function toggleFlash() {
    app.toggleFlash();
}

function toggleTimer() {
    app.toggleTimer();
}

function closeAnalysis() {
    app.closeAnalysis();
}

function speakAnalysis() {
    app.speakAnalysis();
}

function clearAllData() {
    app.clearAllData();
}

// Initialize the app
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new GeminiCameraApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause camera when page is hidden
        if (app && app.currentStream) {
            app.currentStream.getTracks().forEach(track => track.enabled = false);
        }
    } else {
        // Resume camera when page is visible
        if (app && app.currentStream) {
            app.currentStream.getTracks().forEach(track => track.enabled = true);
        }
    }
});