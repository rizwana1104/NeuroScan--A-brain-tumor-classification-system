const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const resultClass = document.getElementById('resultClass');
const confidenceBadge = document.getElementById('confidenceBadge');
const loading = document.getElementById('loading');

let selectedFile = null;

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        predictBtn.disabled = false;
        resultSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

removeImageBtn.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    previewContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    predictBtn.disabled = true;
    resultSection.classList.add('hidden');
});

const explanationText = document.getElementById('explanationText');

function generateExplanation(className, confidence) {
    const confidencePercent = (confidence * 100).toFixed(2);
    let confidenceLevel = "";

    if (confidence > 0.9) {
        confidenceLevel = "very high confidence";
    } else if (confidence > 0.75) {
        confidenceLevel = "high confidence";
    } else if (confidence > 0.5) {
        confidenceLevel = "moderate confidence";
    } else {
        confidenceLevel = "low confidence";
    }

    let explanation = `The model has identified patterns consistent with <strong>${className}</strong> with <strong>${confidenceLevel} (${confidencePercent}%)</strong>. `;

    explanation += `This prediction is based on visual features extracted from the MRI scan, such as texture, intensity gradients, and structural anomalies that are statistically correlated with this tumor type in our training dataset. `;

    explanation += `<br><br><em style="font-size: 0.8em; color: var(--text-secondary);">Note: This is an AI-generated analysis and should not be used as a substitute for professional medical diagnosis.</em>`;

    return explanation;
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    loading.classList.remove('hidden');
    resultSection.classList.add('hidden');
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        resultClass.textContent = data.class;
        confidenceBadge.textContent = (data.confidence * 100).toFixed(2) + '%';

        // Generate and display explanation
        explanationText.innerHTML = generateExplanation(data.class, data.confidence);

        resultSection.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the image.');
    } finally {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

// Medical Disclaimer Modal Logic
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('disclaimerModal');
    const acceptBtn = document.getElementById('acceptDisclaimer');

    // Show modal on load
    // Using setTimeout to ensure transition works
    setTimeout(() => {
        modal.classList.add('active');
    }, 500);

    // Handle acceptance
    acceptBtn.addEventListener('click', () => {
        modal.classList.remove('active');
        // Optional: Store acceptance in localStorage if we want to show it only once
        // localStorage.setItem('medicalDisclaimerAccepted', 'true');
    });
});
