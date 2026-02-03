// Use centralized config, fallback to localhost if not available
const API_BASE = window.API_CONFIG?.BASE_URL || 'http://localhost:8000';

// DOM Elements
const databaseSelect = document.getElementById('databaseSelect');
const catalogSelect = document.getElementById('catalogSelect');
const inputMethodRadios = document.querySelectorAll('input[name="inputMethod"]');
const manualInputSection = document.getElementById('manualInputSection');
const fileUploadSection = document.getElementById('fileUploadSection');
const manualQueries = document.getElementById('manualQueries');
const trueSqlInput = document.getElementById('trueSqlInput');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const downloadTemplateBtn = document.getElementById('downloadTemplateBtn');
const downloadTemplateExcelBtn = document.getElementById('downloadTemplateExcelBtn');
const baselineSection = document.getElementById('baselineSection');
const baselineModelSelect = document.getElementById('baselineModelSelect');
const temperatureInput = document.getElementById('temperatureInput');
const testRequestBtn = document.getElementById('testRequestBtn');
const runEvaluationBtn = document.getElementById('runEvaluationBtn');
const loading = document.getElementById('loading');
const loadingMessage = document.getElementById('loadingMessage');
const loadingProgress = document.getElementById('loadingProgress');
const progressDetails = document.getElementById('progressDetails');
const resultsSection = document.getElementById('resultsSection');
const resultsTableBody = document.getElementById('resultsTableBody');
const downloadResultsBtn = document.getElementById('downloadResultsBtn');
const downloadDetailedBtn = document.getElementById('downloadDetailedBtn');
const detailedResultsContent = document.getElementById('detailedResultsContent');
const historySection = document.getElementById('historySection');
const historyList = document.getElementById('historyList');

// Model selection containers
const openaiModels = document.getElementById('openaiModels');
const openrouterModels = document.getElementById('openrouterModels');
const ollamaModels = document.getElementById('ollamaModels');

// State
let uploadedFile = null;
let evaluationResults = null;
let allCatalogs = [];

// Event Listeners
inputMethodRadios.forEach(radio => {
    radio.addEventListener('change', handleInputMethodChange);
});
fileInput.addEventListener('change', handleFileSelect);
downloadTemplateBtn.addEventListener('click', () => downloadTemplate('csv'));
downloadTemplateExcelBtn.addEventListener('click', () => downloadTemplate('excel'));
testRequestBtn.addEventListener('click', testEvaluationRequest);
runEvaluationBtn.addEventListener('click', runEvaluation);
downloadResultsBtn.addEventListener('click', downloadResults);
downloadDetailedBtn.addEventListener('click', downloadDetailedReport);
databaseSelect.addEventListener('change', handleDatabaseChange);

// Initialize
async function initializeApp() {
    try {
        await loadDatabases();
        await loadAllModels();
        await loadEvaluationHistory();
        // Initialize baseline section visibility
        checkBaselineVisibility();
    } catch (error) {
        console.error('Initialization error:', error);
        showAlert('Failed to initialize application', 'error');
    }
}

// Load databases
async function loadDatabases() {
    try {
        const response = await fetch(`${API_BASE}/api/databases`);
        const data = await response.json();
        
        if (data.success && data.databases.length > 0) {
            databaseSelect.innerHTML = '';
            data.databases.forEach(db => {
                const option = document.createElement('option');
                option.value = db.value;
                option.textContent = db.label;
                databaseSelect.appendChild(option);
            });
            
            await loadCatalogs(data.databases[0].value);
        }
    } catch (error) {
        console.error('Error loading databases:', error);
        showAlert('Failed to load databases', 'error');
    }
}

// Load catalogs
async function loadCatalogs(database) {
    catalogSelect.innerHTML = '<option value="">Loading...</option>';
    catalogSelect.disabled = true;
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const response = await fetch(`${API_BASE}/api/catalogs?database=${encodeURIComponent(database)}`, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        const data = await response.json();
        
        if (data.success && data.catalogs.length > 0) {
            allCatalogs = data.catalogs;
            catalogSelect.innerHTML = '';
            allCatalogs.forEach(catalog => {
                const option = document.createElement('option');
                option.value = catalog.value;
                option.textContent = catalog.label;
                catalogSelect.appendChild(option);
            });
            catalogSelect.disabled = false;
        } else {
            setDefaultCatalog();
            catalogSelect.disabled = false;
        }
    } catch (error) {
        console.error('Error loading catalogs:', error);
        setDefaultCatalog();
        catalogSelect.disabled = false;
    }
}

function setDefaultCatalog() {
    catalogSelect.innerHTML = '';
    const option = document.createElement('option');
    option.value = 'salespoint_production';
    option.textContent = 'salespoint_production (default)';
    catalogSelect.appendChild(option);
}

// Handle database change
async function handleDatabaseChange() {
    const database = databaseSelect.value;
    await loadCatalogs(database);
}

// Load all models
async function loadAllModels() {
    try {
        // Load OpenAI models
        const openaiResponse = await fetch(`${API_BASE}/api/models/openai`);
        const openaiData = await openaiResponse.json();
        if (openaiData.success) {
            renderModelCheckboxes(openaiModels, openaiData.models, 'openai');
            updateBaselineOptions(openaiData.models);
        }

        // Load OpenRouter models
        const openrouterResponse = await fetch(`${API_BASE}/api/models/openrouter`);
        const openrouterData = await openrouterResponse.json();
        if (openrouterData.success) {
            renderModelCheckboxes(openrouterModels, openrouterData.models, 'openrouter');
            updateBaselineOptions(openrouterData.models);
        }

        // Load Ollama models
        const ollamaResponse = await fetch(`${API_BASE}/api/models/ollama`);
        const ollamaData = await ollamaResponse.json();
        if (ollamaData.success) {
            renderModelCheckboxes(ollamaModels, ollamaData.models, 'ollama');
            updateBaselineOptions(ollamaData.models);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showAlert('Failed to load models', 'error');
    }
}

// Render model checkboxes
function renderModelCheckboxes(container, models, provider) {
    container.innerHTML = '';
    models.forEach(model => {
        const label = document.createElement('label');
        label.className = 'model-checkbox-label';
        label.innerHTML = `
            <input type="checkbox" value="${model.value}" data-provider="${provider}">
            <span>${model.label}</span>
        `;
        container.appendChild(label);
    });
}

// Update baseline options
function updateBaselineOptions(models) {
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.value;
        option.textContent = model.label;
        baselineModelSelect.appendChild(option);
    });
}

// Handle input method change
function handleInputMethodChange(e) {
    if (e.target.value === 'manual') {
        manualInputSection.classList.remove('hidden');
        fileUploadSection.classList.add('hidden');
        uploadedFile = null;
        fileInfo.textContent = '';
        // Show baseline section for manual input initially
        checkBaselineVisibility();
    } else {
        manualInputSection.classList.add('hidden');
        fileUploadSection.classList.remove('hidden');
        baselineSection.classList.add('hidden');
    }
}

// Check if baseline section should be visible
function checkBaselineVisibility() {
    const trueSqlText = trueSqlInput.value.trim();
    if (!trueSqlText) {
        baselineSection.classList.remove('hidden');
    } else {
        baselineSection.classList.add('hidden');
    }
}

// Add event listener for true SQL input
trueSqlInput.addEventListener('input', checkBaselineVisibility);

// Handle file select
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadedFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
        fileInfo.style.color = 'var(--success)';
    }
}

// Download template
async function downloadTemplate(format) {
    try {
        const response = await fetch(`${API_BASE}/api/evaluation/template?format=${format}`);
        const blob = await response.blob();
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evaluation_template.${format === 'excel' ? 'xlsx' : 'csv'}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showAlert('Template downloaded successfully!', 'success');
    } catch (error) {
        console.error('Error downloading template:', error);
        showAlert('Failed to download template', 'error');
    }
}

// Test evaluation request
async function testEvaluationRequest() {
    // Get selected models
    const selectedModels = getSelectedModels();
    if (selectedModels.length === 0) {
        showAlert('Please select at least one model to evaluate', 'error');
        return;
    }

    // Get queries
    let queries = [];
    let trueSqls = [];
    const inputMethod = document.querySelector('input[name="inputMethod"]:checked').value;
    
    if (inputMethod === 'manual') {
        const queryText = manualQueries.value.trim();
        if (!queryText) {
            showAlert('Please enter at least one query', 'error');
            return;
        }
        queries = queryText.split('\n').filter(q => q.trim());
        
        const trueSqlText = trueSqlInput.value.trim();
        if (trueSqlText) {
            trueSqls = trueSqlText.split('\n').map(s => s.trim()).filter(s => s.length > 0);
        }
    } else {
        showAlert('File upload testing not supported in test mode', 'warning');
        return;
    }

    // Check if we need baseline model
    const hasTrueSql = trueSqls.length > 0;
    let baselineModel = null;
    
    if (!hasTrueSql) {
        baselineSection.classList.remove('hidden');
        baselineModel = baselineModelSelect.value;
        if (!baselineModel) {
            showAlert('Please select a baseline model (no true SQL provided)', 'error');
            return;
        }
    }

    const requestData = {
        queries: queries,
        true_sqls: trueSqls || [],
        models: selectedModels,
        baseline_model: baselineModel || null,
        database: databaseSelect.value,
        catalog: catalogSelect.value,
        temperature: parseFloat(temperatureInput.value) || 0.3,
        request_limit: 5,
        iteration_limit: 5
    };

    console.log('=== TEST REQUEST ===');
    console.log('Request data:', JSON.stringify(requestData, null, 2));
    
    try {
        showLoading('Testing request validation...');
        
        const response = await fetch(`${API_BASE}/api/evaluation/test`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        hideLoading();

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            console.error('Test failed:', errorData);
            showAlert(`Test failed: ${JSON.stringify(errorData, null, 2)}`, 'error');
            return;
        }

        const result = await response.json();
        console.log('Test result:', result);
        showAlert(`✅ Test passed! Request is valid. Received: ${JSON.stringify(result.received, null, 2)}`, 'success');
    } catch (error) {
        hideLoading();
        console.error('Test error:', error);
        showAlert(`Test error: ${error.message}`, 'error');
    }
}

// Run evaluation
async function runEvaluation() {
    // Get selected models
    const selectedModels = getSelectedModels();
    if (selectedModels.length === 0) {
        showAlert('Please select at least one model to evaluate', 'error');
        return;
    }

    // Get queries
    let queries = [];
    let trueSqls = [];
    const inputMethod = document.querySelector('input[name="inputMethod"]:checked').value;
    
    if (inputMethod === 'manual') {
        const queryText = manualQueries.value.trim();
        if (!queryText) {
            showAlert('Please enter at least one query', 'error');
            return;
        }
        queries = queryText.split('\n').filter(q => q.trim());
        
        const trueSqlText = trueSqlInput.value.trim();
        if (trueSqlText) {
            trueSqls = trueSqlText.split('\n').map(s => s.trim()).filter(s => s.length > 0);
        }
        
        console.log('Manual input - Queries:', queries.length, 'True SQLs:', trueSqls.length);
    } else {
        if (!uploadedFile) {
            showAlert('Please upload a file', 'error');
            return;
        }
    }

    // Show loading
    showLoading('Preparing evaluation...');
    resultsSection.classList.add('hidden');
    
    let progressInterval = null;

    try {
        let evaluationData;
        
        if (inputMethod === 'file') {
            // Upload file first
            const formData = new FormData();
            formData.append('file', uploadedFile);
            
            const uploadResponse = await fetch(`${API_BASE}/api/evaluation/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!uploadResponse.ok) {
                throw new Error('File upload failed');
            }
            
            const uploadData = await uploadResponse.json();
            // Filter out null/empty true_sqls
            const filteredTrueSqls = uploadData.true_sqls 
                ? uploadData.true_sqls.filter(sql => sql && sql.trim().length > 0)
                : [];
            evaluationData = {
                queries: uploadData.queries,
                true_sqls: filteredTrueSqls
            };
        } else {
            evaluationData = {
                queries: queries,
                true_sqls: trueSqls
            };
        }

        // Check if we need baseline model (after we have the data)
        const hasTrueSql = evaluationData.true_sqls.length > 0;
        let baselineModel = null;
        
        // Show baseline section if no true SQL provided
        if (!hasTrueSql) {
            hideLoading();
            baselineSection.classList.remove('hidden');
            baselineModel = baselineModelSelect.value;
            if (!baselineModel) {
                showAlert('Please select a baseline model (no true SQL provided)', 'error');
                return;
            }
            showLoading('Preparing evaluation...');
        } else {
            baselineSection.classList.add('hidden');
        }

        // Start evaluation
        // Filter out null/empty true_sqls to avoid validation errors
        const cleanTrueSqls = (evaluationData.true_sqls || [])
            .filter(sql => sql !== null && sql !== undefined && sql.trim && sql.trim().length > 0);
        
        const requestData = {
            queries: evaluationData.queries,
            true_sqls: cleanTrueSqls.length > 0 ? cleanTrueSqls : [],
            models: selectedModels,
            baseline_model: baselineModel || null,
            database: databaseSelect.value,
            catalog: catalogSelect.value,
            temperature: parseFloat(temperatureInput.value) || 0.3,
            request_limit: 5,
            iteration_limit: 5
        };

        console.log('=== EVALUATION REQUEST DEBUG ===');
        console.log('Queries count:', requestData.queries.length);
        console.log('Queries:', requestData.queries);
        console.log('True SQLs count:', requestData.true_sqls.length);
        console.log('True SQLs:', requestData.true_sqls);
        console.log('Models count:', requestData.models.length);
        console.log('Models:', requestData.models);
        console.log('Baseline model:', requestData.baseline_model);
        console.log('Database:', requestData.database);
        console.log('Catalog:', requestData.catalog);
        console.log('Temperature:', requestData.temperature);
        console.log('Full request:', JSON.stringify(requestData, null, 2));
        console.log('=== END DEBUG ===');
        
        updateLoadingMessage('Running evaluation... This may take several minutes.');
        
        // Simulate progress updates (since we can't get real-time updates without WebSockets)
        const totalQueries = requestData.queries.length;
        const totalModels = requestData.models.length;
        let completedCount = 0;
        const totalSteps = totalQueries * totalModels;
        
        // Update progress periodically
        progressInterval = setInterval(() => {
            if (completedCount < totalSteps) {
                completedCount++;
                const queryIndex = Math.floor(completedCount / totalModels);
                const modelIndex = completedCount % totalModels;
                updateProgressUI(
                    queryIndex,
                    totalQueries,
                    modelIndex,
                    totalModels,
                    requestData.queries[Math.min(queryIndex, totalQueries - 1)],
                    requestData.models[Math.min(modelIndex, totalModels - 1)].value
                );
            }
        }, 2000); // Update every 2 seconds
        
        const response = await fetch(`${API_BASE}/api/evaluation/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        clearInterval(progressInterval);

        if (!response.ok) {
            let errorMessage = 'Evaluation failed';
            try {
                const errorData = await response.json();
                console.error('Server error response:', errorData);
                errorMessage = errorData.detail || JSON.stringify(errorData);
            } catch (e) {
                const errorText = await response.text();
                console.error('Server error text:', errorText);
                errorMessage = errorText || `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }

        evaluationResults = await response.json();
        hideLoading();
        displayResults(evaluationResults);
        
        showAlert('Evaluation completed successfully!', 'success');
        
        // Reload history after completing evaluation
        await loadEvaluationHistory();
    } catch (error) {
        console.error('Evaluation error:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack
        });
        
        // Clear progress interval on error
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        hideLoading();
        
        // Better error message formatting
        let errorMsg = 'Evaluation failed';
        if (error.message) {
            errorMsg = error.message;
        }
        if (typeof errorMsg === 'object') {
            errorMsg = JSON.stringify(errorMsg, null, 2);
        }
        
        showAlert(`Evaluation failed: ${errorMsg}`, 'error');
    }
}

// Get selected models
function getSelectedModels() {
    const checkboxes = document.querySelectorAll('.model-checkbox-label input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => ({
        value: cb.value,
        provider: cb.dataset.provider
    }));
}

// Display results
function displayResults(results) {
    // Display summary table
    resultsTableBody.innerHTML = '';
    results.summary.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${escapeHtml(item.provider.toUpperCase())}</td>
            <td>${escapeHtml(item.model_name)}</td>
            <td>${item.accuracy.toFixed(2)}%</td>
            <td>${formatTime(item.avg_speed)}</td>
            <td>${formatTime(item.total_time)}</td>
            <td>${item.avg_iterations.toFixed(1)}</td>
            <td>${item.temperature.toFixed(1)}</td>
            <td>${item.success_rate.toFixed(1)}%</td>
        `;
        resultsTableBody.appendChild(row);
    });

    // Display detailed results
    displayDetailedResults(results.detailed);

    // Create visualizations
    createCharts(results.summary);

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display detailed results
function displayDetailedResults(detailed) {
    detailedResultsContent.innerHTML = '';
    
    detailed.forEach((queryResult, index) => {
        const queryCard = document.createElement('div');
        queryCard.className = 'query-result-card';
        queryCard.innerHTML = `
            <h5>Query ${index + 1}: ${escapeHtml(queryResult.query)}</h5>
            ${queryResult.true_sql ? `<div class="true-sql"><strong>True SQL:</strong><pre>${escapeHtml(queryResult.true_sql)}</pre></div>` : ''}
            <div class="model-results">
                ${queryResult.results.map(r => `
                    <div class="model-result-item">
                        <div class="model-result-header">
                            <strong>${escapeHtml(r.model)}</strong>
                            <span class="${r.success ? 'success-badge' : 'error-badge'}">
                                ${r.success ? '✓' : '✗'}
                            </span>
                        </div>
                        <div class="model-result-metrics">
                            <span>⏱️ ${formatTime(r.generation_time)}</span>
                            <span>🎯 Accuracy: ${r.accuracy_score.toFixed(2)}%</span>
                            <span>🔄 ${r.iterations} iterations</span>
                        </div>
                        ${r.generated_sql ? `<pre class="generated-sql">${escapeHtml(r.generated_sql)}</pre>` : ''}
                        ${r.error ? `<div class="error-message">${escapeHtml(r.error)}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
        detailedResultsContent.appendChild(queryCard);
    });
}

// Create charts
function createCharts(summary) {
    const labels = summary.map(s => s.model_name);
    const accuracyData = summary.map(s => s.accuracy);
    const speedData = summary.map(s => s.avg_speed);
    const colors = [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(139, 92, 246, 0.8)',
        'rgba(236, 72, 153, 0.8)',
        'rgba(20, 184, 166, 0.8)',
        'rgba(251, 146, 60, 0.8)'
    ];

    // Grouped Bar Chart (Accuracy & Speed)
    const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(comparisonCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracyData,
                    backgroundColor: 'rgba(59, 130, 246, 0.8)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Speed (seconds)',
                    data: speedData,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#e5e7eb' }
                },
                datalabels: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });

    // Accuracy Bar Chart
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    new Chart(accuracyCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: accuracyData,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                datalabels: {
                    anchor: 'end',
                    align: 'top',
                    color: '#e5e7eb',
                    formatter: (value) => value.toFixed(1) + '%'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });

    // Speed Bar Chart
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    new Chart(speedCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Speed (seconds)',
                data: speedData,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                datalabels: {
                    anchor: 'end',
                    align: 'top',
                    color: '#e5e7eb',
                    formatter: (value) => formatTime(value)
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });

    // Heatmap (using matrix visualization)
    const heatmapCtx = document.getElementById('heatmapChart').getContext('2d');
    const heatmapData = summary.map((s, idx) => ({
        x: idx,
        y: 0,
        v: s.accuracy
    })).concat(summary.map((s, idx) => ({
        x: idx,
        y: 1,
        v: (s.avg_speed / Math.max(...speedData)) * 100
    })));

    new Chart(heatmapCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracyData,
                    backgroundColor: accuracyData.map(v => getHeatColor(v)),
                    borderWidth: 1
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                datalabels: {
                    color: '#111827',
                    font: { weight: 'bold' },
                    formatter: (value) => value.toFixed(1) + '%'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    ticks: { color: '#e5e7eb' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// Get heat color based on value
function getHeatColor(value) {
    if (value >= 90) return 'rgba(16, 185, 129, 0.9)'; // Green
    if (value >= 75) return 'rgba(59, 130, 246, 0.9)'; // Blue
    if (value >= 60) return 'rgba(245, 158, 11, 0.9)'; // Orange
    return 'rgba(239, 68, 68, 0.9)'; // Red
}

// Download results
function downloadResults() {
    if (!evaluationResults) return;
    
    const csv = convertToCSV(evaluationResults.summary);
    downloadFile(csv, 'evaluation_results.csv', 'text/csv');
    showAlert('Results downloaded successfully!', 'success');
}

// Download detailed report
function downloadDetailedReport() {
    if (!evaluationResults) return;
    
    let report = 'SQL Generation Model Evaluation Report\n';
    report += '='.repeat(60) + '\n\n';
    
    report += 'Summary:\n';
    report += '-'.repeat(60) + '\n';
    evaluationResults.summary.forEach(s => {
        report += `Model: ${s.model_name} (${s.provider})\n`;
        report += `  Accuracy: ${s.accuracy.toFixed(2)}%\n`;
        report += `  Avg Speed: ${formatTime(s.avg_speed)}\n`;
        report += `  Success Rate: ${s.success_rate.toFixed(1)}%\n`;
        report += `  Avg Iterations: ${s.avg_iterations.toFixed(1)}\n`;
        report += `  Temperature: ${s.temperature.toFixed(1)}\n\n`;
    });
    
    report += '\n\nDetailed Results:\n';
    report += '='.repeat(60) + '\n';
    evaluationResults.detailed.forEach((qr, idx) => {
        report += `\nQuery ${idx + 1}: ${qr.query}\n`;
        report += '-'.repeat(60) + '\n';
        if (qr.true_sql) {
            report += `True SQL: ${qr.true_sql}\n\n`;
        }
        qr.results.forEach(r => {
            report += `  Model: ${r.model}\n`;
            report += `    Success: ${r.success ? 'Yes' : 'No'}\n`;
            report += `    Accuracy: ${r.accuracy_score.toFixed(2)}%\n`;
            report += `    Speed: ${formatTime(r.generation_time)}\n`;
            report += `    Iterations: ${r.iterations}\n`;
            if (r.generated_sql) {
                report += `    Generated SQL: ${r.generated_sql}\n`;
            }
            if (r.error) {
                report += `    Error: ${r.error}\n`;
            }
            report += '\n';
        });
    });
    
    downloadFile(report, 'evaluation_detailed_report.txt', 'text/plain');
    showAlert('Detailed report downloaded successfully!', 'success');
}

// Convert to CSV
function convertToCSV(data) {
    const headers = ['Provider', 'Model', 'Accuracy (%)', 'Avg Speed (s)', 'Total Time (s)', 'Avg Iterations', 'Temperature', 'Success Rate (%)'];
    const rows = data.map(item => [
        item.provider,
        item.model_name,
        item.accuracy.toFixed(2),
        item.avg_speed.toFixed(2),
        item.total_time.toFixed(2),
        item.avg_iterations.toFixed(1),
        item.temperature.toFixed(1),
        item.success_rate.toFixed(1)
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

// Download file
function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// Utility functions
function showLoading(message) {
    loadingMessage.textContent = message;
    loadingProgress.textContent = '';
    loading.classList.remove('hidden');
}

function updateLoadingMessage(message, progress = '') {
    loadingMessage.textContent = message;
    loadingProgress.textContent = progress;
}

function hideLoading() {
    loading.classList.add('hidden');
}

function showAlert(message, type = 'error') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alert.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000; max-width: 400px;';
    
    document.body.appendChild(alert);
    setTimeout(() => alert.remove(), 5000);
}

function formatTime(seconds) {
    if (!seconds) return '00:00';
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// Load evaluation history
async function loadEvaluationHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/evaluation/saved`);
        const data = await response.json();
        
        if (data.success && data.evaluations.length > 0) {
            historyList.innerHTML = '';
            data.evaluations.forEach(eval => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.onclick = () => loadPreviousEvaluation(eval.id);
                
                const date = new Date(eval.timestamp);
                const formattedDate = date.toLocaleString();
                
                historyItem.innerHTML = `
                    <div class="history-item-header">
                        <span class="history-item-title">📊 Evaluation ${eval.id}</span>
                        <span class="history-item-date">${formattedDate}</span>
                    </div>
                    <div class="history-item-details">
                        <span class="history-item-badge">💾 ${eval.database}</span>
                        <span class="history-item-badge">📚 ${eval.catalog}</span>
                        <span class="history-item-badge">🤖 ${eval.models_count} models</span>
                        <span class="history-item-badge">📝 ${eval.queries_count} queries</span>
                    </div>
                `;
                historyList.appendChild(historyItem);
            });
        } else {
            historyList.innerHTML = '<p class="text-secondary">No previous evaluations found.</p>';
        }
    } catch (error) {
        console.error('Error loading evaluation history:', error);
        historyList.innerHTML = '<p class="text-secondary">Failed to load evaluation history.</p>';
    }
}

// Load previous evaluation
async function loadPreviousEvaluation(evaluationId) {
    try {
        showLoading('Loading previous evaluation...');
        
        const response = await fetch(`${API_BASE}/api/evaluation/saved/${evaluationId}`);
        const data = await response.json();
        
        if (data.success && data.results) {
            evaluationResults = data.results;
            hideLoading();
            displayResults(evaluationResults);
            showAlert('Previous evaluation loaded successfully!', 'success');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error('Invalid response');
        }
    } catch (error) {
        console.error('Error loading previous evaluation:', error);
        hideLoading();
        showAlert('Failed to load previous evaluation', 'error');
    }
}

// Update progress during evaluation
function updateProgressUI(queriesCompleted, totalQueries, modelsCompleted, totalModels, currentQuery, currentModel) {
    const progressText = `Processing Query ${queriesCompleted + 1}/${totalQueries} - Model ${modelsCompleted + 1}/${totalModels}`;
    loadingProgress.textContent = progressText;
    
    // Update detailed progress
    if (progressDetails) {
        const progressHTML = `
            <div class="progress-item active">
                <span class="progress-status">⚡</span>
                <div>
                    <strong>${currentModel}</strong>
                    <div class="text-secondary">${currentQuery.substring(0, 60)}${currentQuery.length > 60 ? '...' : ''}</div>
                </div>
            </div>
        `;
        progressDetails.innerHTML = progressHTML;
    }
}

// Initialize on load
window.addEventListener('load', initializeApp);

