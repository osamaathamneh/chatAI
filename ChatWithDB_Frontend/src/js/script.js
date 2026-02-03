// Use centralized config, fallback to localhost if not available
const API_BASE = window.API_CONFIG?.BASE_URL || 'http://localhost:8000';

// DOM Elements
const queryInput = document.getElementById('query');
const searchBtn = document.getElementById('searchBtn');
const generateBtn = document.getElementById('generateBtn');
const executeBtn = document.getElementById('executeBtn');
const copyBtn = document.getElementById('copyBtn');
const clearSqlBtn = document.getElementById('clearSqlBtn');
const clearExecutionBtn = document.getElementById('clearExecutionBtn');
const loading = document.getElementById('loading');
const loadingMessage = document.getElementById('loadingMessage');
const searchResults = document.getElementById('searchResults');
const sqlResults = document.getElementById('sqlResults');
const executionResults = document.getElementById('executionResults');

// Configuration Elements
const providerSelect = document.getElementById('providerSelect');
const databaseSelect = document.getElementById('databaseSelect');
const catalogSelect = document.getElementById('catalogSelect');
const modelSelect = document.getElementById('modelSelect');
const requestLimitInput = document.getElementById('requestLimit');
const iterationLimitInput = document.getElementById('iterationLimit');
const temperatureInput = document.getElementById('temperature');
const searchLimitInput = document.getElementById('searchLimit');

// History Elements
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyList = document.getElementById('historyList');
const examplesGrid = document.getElementById('examplesGrid');
const examplesSource = document.getElementById('examplesSource');

let currentSQL = '';
let allCatalogs = [];

// Event Listeners
searchBtn.addEventListener('click', handleSearch);
generateBtn.addEventListener('click', handleGenerateSQL);
executeBtn.addEventListener('click', handleExecuteSQL);
copyBtn.addEventListener('click', copyToClipboard);
clearSqlBtn.addEventListener('click', clearSQLResults);
clearExecutionBtn.addEventListener('click', clearExecutionResults);
providerSelect.addEventListener('change', handleProviderChange);
databaseSelect.addEventListener('change', handleDatabaseChange);
refreshHistoryBtn.addEventListener('click', loadHistory);
clearHistoryBtn.addEventListener('click', clearHistory);

// Handle Enter key in textarea (Ctrl+Enter to generate)
queryInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        handleGenerateSQL();
    }
});

// Initialize on page load
async function initializeApp() {
    try {
        // Load databases
        await loadDatabases();
        
        // Load models based on default provider
        await loadModels('openai');
        
        // Load history and examples
        await loadHistory();
        await loadExamples();
    } catch (error) {
        console.error('Initialization error:', error);
        showAlert('Failed to initialize application');
    }
}

// Load available databases
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
            
            // Load catalogs for the first database
            await loadCatalogs(data.databases[0].value);
        }
    } catch (error) {
        console.error('Error loading databases:', error);
        showAlert('Failed to load databases');
    }
}

// Load semantic catalogs for a database
async function loadCatalogs(database) {
    // Show loading state
    catalogSelect.innerHTML = '<option value="">Loading catalogs...</option>';
    catalogSelect.disabled = true;
    
    try {
        // Add timeout to prevent infinite loading
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const response = await fetch(`${API_BASE}/api/catalogs?database=${encodeURIComponent(database)}`, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        const data = await response.json();
        
        if (data.success && data.catalogs.length > 0) {
            allCatalogs = data.catalogs;
            updateCatalogDropdown();
            catalogSelect.disabled = false;
        } else {
            // No catalogs found
            setDefaultCatalog();
            catalogSelect.disabled = false;
        }
    } catch (error) {
        console.error('Error loading catalogs:', error);
        // Set default catalog on failure
        setDefaultCatalog();
        catalogSelect.disabled = false;
    }
}

// Set default catalog when API fails
function setDefaultCatalog() {
    catalogSelect.innerHTML = '';
    const option = document.createElement('option');
    option.value = 'salespoint_production';
    option.textContent = 'salespoint_production (default)';
    catalogSelect.appendChild(option);
}

// Update catalog dropdown based on selected database
function updateCatalogDropdown() {
    catalogSelect.innerHTML = '';
    
    if (!allCatalogs || allCatalogs.length === 0) {
        // Set default catalog if no catalogs found
        setDefaultCatalog();
        return;
    }
    
    // Show all available catalogs from the API response
    allCatalogs.forEach(catalog => {
        const option = document.createElement('option');
        option.value = catalog.value;
        option.textContent = catalog.label;
        catalogSelect.appendChild(option);
    });
}

// Load models based on provider
async function loadModels(provider) {
    try {
        let endpoint;
        if (provider === 'ollama') {
            endpoint = '/api/models/ollama';
        } else if (provider === 'openrouter') {
            endpoint = '/api/models/openrouter';
        } else {
            endpoint = '/api/models/openai';
        }
        const response = await fetch(`${API_BASE}${endpoint}`);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}: ${response.statusText}` }));
            throw new Error(errorData.detail || `Failed to load models: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.detail || 'Failed to load models');
        }
        
        if (data.models && data.models.length > 0) {
            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.value;
                option.textContent = model.label;
                modelSelect.appendChild(option);
            });
            
            // Set default model based on provider
            if (provider === 'openai') {
                // First model should be GPT-5 after sorting
                if (data.models.length > 0) {
                    modelSelect.value = data.models[0].value;
                }
                // Double-check and force GPT-5 if available
                const gpt5Option = Array.from(modelSelect.options).find(opt => 
                    opt.value === 'openai:gpt-5' || opt.value.includes('gpt-5')
                );
                if (gpt5Option) {
                    modelSelect.value = gpt5Option.value;
                    console.log('Selected default model: GPT-5');
                }
            } else if (provider === 'openrouter') {
                // For OpenRouter, select gpt-oss-20b if available (default), otherwise first model
                const gptOss20bOption = Array.from(modelSelect.options).find(opt => 
                    opt.value.includes('gpt-oss-20b') || opt.value.includes('openai/gpt-oss-20b')
                );
                if (gptOss20bOption) {
                    modelSelect.value = gptOss20bOption.value;
                    console.log('Selected default OpenRouter model: GPT OSS 20B');
                } else if (data.models.length > 0) {
                    modelSelect.value = data.models[0].value;
                    console.log('Selected default OpenRouter model:', data.models[0].label);
                }
            } else {    
                // For Ollama, select first model
                if (data.models.length > 0) {
                    modelSelect.value = data.models[0].value;
                }
            }
        } else {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            if (provider === 'ollama') {
                showAlert('No Ollama models found. Make sure Ollama is running.', 'warning');
            }
        }
    } catch (error) {
        console.error('Error loading models:', error);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        const errorMessage = error.message || `Failed to load ${provider} models`;
        showAlert(errorMessage, 'error');
    }
}

// Handle provider change
async function handleProviderChange() {
    const provider = providerSelect.value;
    await loadModels(provider);
}

// Handle database change
async function handleDatabaseChange() {
    const database = databaseSelect.value;
    await loadCatalogs(database);
    await loadExamples(database);
    await loadHistory(database);
}

// Set example query
function setQuery(text) {
    queryInput.value = text.trim();
    queryInput.focus();
}

// Show/Hide Loading
function showLoading(message = 'Processing your request...') {
    loadingMessage.textContent = message;
    loading.classList.remove('hidden');
    searchResults.classList.add('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

// Show Alert
function showAlert(message, type = 'error') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    
    const container = document.querySelector('.card');
    container.insertBefore(alert, container.firstChild);
    
    setTimeout(() => alert.remove(), 5000);
}

// Handle Search
async function handleSearch() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showAlert('Please enter a question');
        return;
    }
    
    showLoading('🔍 Searching semantic catalog...');
    
    try {
        const limit = parseInt(searchLimitInput.value) || 10;
        const catalog_db = databaseSelect.value;
        const catalog_name = catalogSelect.value;
        
        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, limit, catalog_db, catalog_name })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displaySearchResults(data.results);
    } catch (error) {
        console.error('Search error:', error);
        showAlert(`Search failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display Search Results
function displaySearchResults(results) {
    const searchContent = document.getElementById('searchContent');
    
    if (!results || results.length === 0) {
        searchContent.innerHTML = '<p class="text-secondary">No results found.</p>';
    } else {
        searchContent.innerHTML = results.map(result => `
            <div class="search-result-item">
                <div class="item-name">${escapeHtml(result.item || 'Unknown')}</div>
                <div class="item-description">${escapeHtml(result.description || 'No description')}</div>
            </div>
        `).join('');
    }
    
    searchResults.classList.remove('hidden');
}

// Handle Generate SQL
async function handleGenerateSQL() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showAlert('Please enter a question');
        return;
    }
    
    // Clear previous results before generating new SQL
    clearSQLResults();
    clearExecutionResults();
    
    showLoading('✨ Generating SQL query... This may take a moment.');
    
    try {
        const provider = providerSelect.value;
        const model = modelSelect.value;
        const request_limit = parseInt(requestLimitInput.value) || 5;
        const iteration_limit = parseInt(iterationLimitInput.value) || 5;
        const temperature = parseFloat(temperatureInput.value) || 0.3;
        const target_db = databaseSelect.value;
        const catalog_db = databaseSelect.value;
        const catalog_name = catalogSelect.value;
        
        // Debug logging
        console.log('Sending request with parameters:', {
            temperature,
            iteration_limit,
            request_limit,
            provider,
            model
        });
        
        const response = await fetch(`${API_BASE}/api/generate-sql`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query,
                model,
                request_limit,
                iteration_limit,
                temperature,
                target_db,
                catalog_db,
                catalog_name,
                provider
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displaySQLResults(data.sql, data.generation_time, data.actual_iterations, data.token_usage, iteration_limit);
        
        // Reload history and examples after successful generation
        await loadHistory();
        await loadExamples();
    } catch (error) {
        console.error('SQL generation error:', error);
        showAlert(`SQL generation failed: ${error.message}`);
        
        // Still reload history to show the failed attempt
        await loadHistory();
    } finally {
        hideLoading();
    }
}

// Display SQL Results
function displaySQLResults(sql, generationTime = null, iterations = null, tokens = null, iterationLimit = null) {
    currentSQL = sql;
    const sqlQuery = document.getElementById('sqlQuery');
    
    // Build stats display
    let statsHtml = '';
    if (generationTime || iterations || tokens) {
        const timeFormatted = generationTime ? formatTime(generationTime) : null;
        const parts = [];
        
        if (timeFormatted) {
            parts.push(`⏱️ ${timeFormatted}`);
        }
        if (iterations !== null) {
            const iterText = iterationLimit ? `${iterations}/${iterationLimit}` : iterations;
            parts.push(`🔄 ${iterText} iteration${iterations !== 1 ? 's' : ''}`);
        }
        if (tokens !== null && tokens.total_tokens) {
            parts.push(`🎯 ${tokens.total_tokens.toLocaleString()} tokens (${tokens.input_tokens.toLocaleString()} in, ${tokens.output_tokens.toLocaleString()} out)`);
        }
        
        if (parts.length > 0) {
            statsHtml = `<div class="generation-stats">${parts.join(' | ')}</div>`;
        }
    }
    
    // Add warning if iteration limit was reached
    let warningHtml = '';
    if (iterations !== null && iterationLimit !== null && iterations >= iterationLimit) {
        warningHtml = `<div class="alert alert-warning" style="margin-top: 0.5rem; padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; border-radius: 0.25rem;">
            <strong>⚠️ Iteration Limit Reached:</strong> The model used all ${iterationLimit} attempts. The SQL might still have errors. 
            Consider increasing the <strong>Iteration Limit</strong> (try 7-10) or checking the SQL carefully before executing.
        </div>`;
    }
    
    // Display SQL with stats and warnings
    sqlQuery.innerHTML = statsHtml + warningHtml + '<pre>' + escapeHtml(sql) + '</pre>';
    
    sqlResults.classList.remove('hidden');
    executionResults.classList.add('hidden');
    
    // Scroll to SQL results
    sqlResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle Execute SQL
async function handleExecuteSQL() {
    if (!currentSQL) {
        showAlert('No SQL query to execute');
        return;
    }
    
    showLoading('▶️ Executing SQL query...');
    
    try {
        const target_db = databaseSelect.value;
        
        const response = await fetch(`${API_BASE}/api/execute-sql`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sql: currentSQL, target_db })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayExecutionResults(data);
    } catch (error) {
        console.error('SQL execution error:', error);
        showAlert(`SQL execution failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display Execution Results
function displayExecutionResults(data) {
    const resultsContent = document.getElementById('queryResultsContent');
    
    if (!data.rows || data.rows.length === 0) {
        resultsContent.innerHTML = '<p class="text-secondary">Query executed successfully. No rows returned.</p>';
    } else {
        const tableHtml = `
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            ${data.columns.map(col => `<th>${escapeHtml(col)}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.rows.map(row => `
                            <tr>
                                ${data.columns.map(col => `<td>${escapeHtml(String(row[col] ?? ''))}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            <p style="margin-top: 1rem; color: var(--text-secondary);">
                ${data.row_count} row${data.row_count !== 1 ? 's' : ''} returned
            </p>
        `;
        resultsContent.innerHTML = tableHtml;
    }
    
    // Keep SQL visible - don't hide it!
    executionResults.classList.remove('hidden');
    showAlert('Query executed successfully!', 'success');
    
    // Scroll to results
    executionResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Copy to Clipboard
async function copyToClipboard() {
    try {
        await navigator.clipboard.writeText(currentSQL);
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<span class="icon">✓</span> Copied!';
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
        }, 2000);
    } catch (error) {
        showAlert('Failed to copy to clipboard');
    }
}

// Load query history
async function loadHistory(database = null) {
    try {
        const db = database || databaseSelect.value;
        const url = db ? `${API_BASE}/api/history?database=${encodeURIComponent(db)}` : `${API_BASE}/api/history`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            displayHistory(data.history);
        } else {
            historyList.innerHTML = '<p class="text-secondary">No query history yet. Generate some SQL to see history!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<p class="text-secondary">Failed to load history</p>';
    }
}

// Format seconds to mm:ss
function formatTime(seconds) {
    if (!seconds) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Display history items
function displayHistory(history) {
    historyList.innerHTML = history.map(item => {
        const timestamp = new Date(item.timestamp).toLocaleString();
        let providerClass = 'provider-openai';
        if (item.provider === 'ollama') {
            providerClass = 'provider-ollama';
        } else if (item.provider === 'openrouter') {
            providerClass = 'provider-openrouter';
        }
        const statusClass = item.success ? 'history-item-success' : 'history-item-error';
        const statusIcon = item.success ? '✓' : '✗';
        const modelName = item.model.replace('openai:', '');
        const timeFormatted = formatTime(item.generation_time);
        
        // Extract database name from URL
        const dbMatch = item.database_url ? item.database_url.match(/\/([^\/]+)$/) : null;
        const dbName = dbMatch ? dbMatch[1] : 'Unknown';
        
        return `
            <div class="history-item" onclick="setQuery('${escapeHtml(item.query).replace(/'/g, "\\'")}')">
                <div class="history-item-header">
                    <div class="history-item-query">${escapeHtml(item.query)}</div>
                    <span class="${statusClass}">${statusIcon}</span>
                </div>
                <div class="history-item-meta">
                    <span class="history-item-time">⏱️ ${timeFormatted}</span>
                    <span class="history-item-provider ${providerClass}">${item.provider.toUpperCase()}</span>
                    <span class="text-secondary">${modelName}</span>
                    <span class="text-secondary">🗄️ ${dbName}</span>
                    <span class="text-secondary">📚 ${item.catalog_name || 'N/A'}</span>
                </div>
                <div class="history-item-meta" style="margin-top: 0.25rem;">
                    <span class="text-secondary">🔄 Iterations: ${item.actual_iterations || 'N/A'}/${item.iteration_limit || 5}</span>
                    <span class="text-secondary">📨 Req Limit: ${item.request_limit || 5}</span>
                    <span class="text-secondary">🌡️ Temp: ${item.temperature !== null && item.temperature !== undefined ? item.temperature.toFixed(1) : 'N/A'}</span>
                    <span class="text-secondary">🎯 Tokens: ${item.total_tokens ? item.total_tokens.toLocaleString() : 'N/A'}</span>
                    <span class="text-secondary">${timestamp}</span>
                </div>
                ${item.generated_sql && item.success ? `<div class="history-item-sql">${escapeHtml(item.generated_sql.substring(0, 100))}${item.generated_sql.length > 100 ? '...' : ''}</div>` : ''}
                ${item.error_message ? `<div class="history-item-error">${escapeHtml(item.error_message)}</div>` : ''}
            </div>
        `;
    }).join('');
}

// Load examples based on database
async function loadExamples(database = null) {
    try {
        const db = database || databaseSelect.value;
        const url = db ? `${API_BASE}/api/examples?database=${encodeURIComponent(db)}` : `${API_BASE}/api/examples`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.examples.length > 0) {
            displayExamples(data.examples, db);
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Display examples
function displayExamples(examples, database) {
    const hasHistory = examples.some(ex => ex.generated_sql);
    const sourceText = hasHistory ? '(from your successful queries)' : '(default examples)';
    examplesSource.textContent = sourceText;
    
    examplesGrid.innerHTML = examples.map(example => `
        <div class="example-card" onclick="setQuery(this.textContent.trim())">
            ${escapeHtml(example.query)}
        </div>
    `).join('');
}

// Clear history
async function clearHistory() {
    // First confirmation
    if (!confirm('⚠️ WARNING: This will permanently delete ALL query history!\n\nThis action CANNOT be undone.\n\nAre you sure you want to continue?')) {
        return;
    }
    
    // Second confirmation (double-check)
    if (!confirm('🔴 FINAL CONFIRMATION\n\nYou are about to delete ALL experiments and query history permanently.\n\nClick OK only if you are absolutely sure.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/history`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.success) {
            showAlert('✅ History cleared successfully!', 'success');
            await loadHistory();
            await loadExamples();
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        showAlert('❌ Failed to clear history');
    }
}

// Clear SQL Results
function clearSQLResults() {
    currentSQL = '';
    sqlResults.classList.add('hidden');
    const sqlQuery = document.getElementById('sqlQuery');
    sqlQuery.innerHTML = '';
}

// Clear Execution Results
function clearExecutionResults() {
    executionResults.classList.add('hidden');
    const queryResultsContent = document.getElementById('queryResultsContent');
    queryResultsContent.innerHTML = '';
}

// Utility: Escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Initialize app on load
window.addEventListener('load', async () => {
    try {
        // Check API health first
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('API Status:', data);
        
        // Initialize the app
        await initializeApp();
    } catch (error) {
        console.error('Initialization failed:', error);
        showAlert('Warning: Could not connect to API');
    }
});