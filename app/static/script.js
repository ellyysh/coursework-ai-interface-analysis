const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsDiv = document.getElementById('results');

let currentFile = null;

uploadArea.addEventListener('click', () => fileInput.click());
selectBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        currentFile = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (ev) => {
            previewImg.src = ev.target.result;
            previewContainer.style.display = 'block';
            uploadArea.style.display = 'none';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(currentFile);
    }
});

clearBtn.addEventListener('click', () => {
    currentFile = null;
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsDiv.classList.add('hidden');
    fileInput.value = '';
});

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    const formData = new FormData();
    formData.append('screenshot', currentFile);

    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('hidden');
    resultsDiv.classList.add('hidden');

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert('Ошибка: ' + (data.error || 'Не удалось проанализировать'));
        }
    } catch (err) {
        alert('Ошибка соединения с сервером');
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.classList.add('hidden');
    }
});

function displayResults(data) {
    document.getElementById('clarityScore').innerText = data.clarity_score.toFixed(1);
    
    const issuesHtml = data.issues.map(issue => `
        <div class="issue-item severity-${issue.severity === 'высокая' ? 'high' : (issue.severity === 'средняя' ? 'medium' : 'low')}">
            <div class="issue-type">⚠️ ${issue.type} (${issue.severity})</div>
            <div class="recommendation">💡 ${issue.recommendation}</div>
        </div>
    `).join('');
    
    document.getElementById('issuesList').innerHTML = issuesHtml;
    document.getElementById('heatmapNote').innerHTML = `<div style="margin-top:1rem;padding:0.75rem;background:#eef2ff;border-radius:1rem;">📊 ${data.heatmap_hint}</div>`;
    document.getElementById('metaInfo').innerHTML = `<small>Анализ выполнен: ${data.analyzed_at}</small>`;
    
    resultsDiv.classList.remove('hidden');
}