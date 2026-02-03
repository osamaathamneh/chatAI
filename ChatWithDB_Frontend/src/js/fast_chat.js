// Use centralized config, fallback to localhost if not available
const API_BASE = window.API_CONFIG?.BASE_URL || 'http://localhost:8000';

// DOM Elements
const queryInput = document.getElementById('query');
const generateBtn = document.getElementById('generateBtn');
const executeBtn = document.getElementById('executeBtn');
const copyBtn = document.getElementById('copyBtn');
const clearSqlBtn = document.getElementById('clearSqlBtn');
const clearExecutionBtn = document.getElementById('clearExecutionBtn');
const loading = document.getElementById('loading');
const loadingMessage = document.getElementById('loadingMessage');
const sqlResults = document.getElementById('sqlResults');
const executionResults = document.getElementById('executionResults');
const templateToggle = document.getElementById('templateToggle');
const templateStatus = document.getElementById('templateStatus');
const templatesList = document.getElementById('templatesList');
const autoExecuteToggle = document.getElementById('autoExecuteToggle');
const autoExecuteStatus = document.getElementById('autoExecuteStatus');
const llmMatchingToggle = document.getElementById('llmMatchingToggle');
const llmMatchingStatus = document.getElementById('llmMatchingStatus');

// Feedback DOM Elements
const likeBtn = document.getElementById('likeBtn');
const dislikeBtn = document.getElementById('dislikeBtn');
const feedbackStatus = document.getElementById('feedbackStatus');
const dislikeModal = document.getElementById('dislikeModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const modalUserQuery = document.getElementById('modalUserQuery');
const modalGeneratedSql = document.getElementById('modalGeneratedSql');
const correctSqlInput = document.getElementById('correctSqlInput');
const regenerateSqlBtn = document.getElementById('regenerateSqlBtn');
const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
const cancelFeedbackBtn = document.getElementById('cancelFeedbackBtn');

// Modal provider and model selects (for regeneration)
const modalProviderSelect = document.getElementById('modalProviderSelect');
const modalModelSelect = document.getElementById('modalModelSelect');

// Configuration Elements
const providerSelect = document.getElementById('providerSelect');
const databaseSelect = document.getElementById('databaseSelect');
const catalogSelect = document.getElementById('catalogSelect');
const modelSelect = document.getElementById('modelSelect');

let currentSQL = '';
let currentUserQuery = '';  // Store current user query for feedback
let currentMethod = '';  // Store generation method (template vs pgai)
let allCatalogs = [];
let templates = [];
let useTemplates = true;
let autoExecute = true;
let useLLMMatching = true;  // ENABLED by default

// Event Listeners
generateBtn.addEventListener('click', handleGenerateSQL);
executeBtn.addEventListener('click', handleExecuteSQL);
copyBtn.addEventListener('click', handleCopySQL);
clearSqlBtn.addEventListener('click', clearSQLResults);
clearExecutionBtn.addEventListener('click', clearExecutionResults);
templateToggle.addEventListener('click', toggleTemplates);
autoExecuteToggle.addEventListener('click', toggleAutoExecute);
llmMatchingToggle.addEventListener('click', toggleLLMMatching);

// Feedback event listeners
if (likeBtn) likeBtn.addEventListener('click', handleLikeFeedback);
if (dislikeBtn) dislikeBtn.addEventListener('click', handleDislikeFeedback);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeDislikeModal);
if (cancelFeedbackBtn) cancelFeedbackBtn.addEventListener('click', closeDislikeModal);
if (regenerateSqlBtn) regenerateSqlBtn.addEventListener('click', handleRegenerateSQL);
if (submitFeedbackBtn) submitFeedbackBtn.addEventListener('click', handleSubmitFeedback);

// Provider and model management
providerSelect.addEventListener('change', async () => {
    await loadModelsForProvider(providerSelect.value);
});

// Modal provider and model management (for regeneration)
if (modalProviderSelect) {
    modalProviderSelect.addEventListener('change', async () => {
        await loadModelsForModalProvider(modalProviderSelect.value);
    });
}

databaseSelect.addEventListener('change', async () => {
    await loadCatalogs(databaseSelect.value);
});

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadDatabases();
    await loadModelsForProvider('openrouter');  // Default to OpenRouter
    await loadTemplates();
});

// Toggle template matching
function toggleTemplates() {
    useTemplates = !useTemplates;
    templateToggle.classList.toggle('active');
    
    if (useTemplates) {
        templateStatus.textContent = 'Enabled';
        templateStatus.style.color = '#667eea';
    } else {
        templateStatus.textContent = 'Disabled';
        templateStatus.style.color = '#cbd5e0';
    }
}

// Toggle auto-execute
function toggleAutoExecute() {
    autoExecute = !autoExecute;
    autoExecuteToggle.classList.toggle('active');
    
    if (autoExecute) {
        autoExecuteStatus.textContent = 'Enabled';
        autoExecuteStatus.style.color = '#667eea';
        generateBtn.textContent = '⚡ Generate & Execute';
    } else {
        autoExecuteStatus.textContent = 'Disabled';
        autoExecuteStatus.style.color = '#cbd5e0';
        generateBtn.textContent = '⚡ Generate SQL';
    }
}

// Toggle LLM matching
function toggleLLMMatching() {
    useLLMMatching = !useLLMMatching;
    llmMatchingToggle.classList.toggle('active');
    
    if (useLLMMatching) {
        llmMatchingStatus.textContent = 'Enabled (🤖 AI)';
        llmMatchingStatus.style.color = '#667eea';
    } else {
        llmMatchingStatus.textContent = 'Disabled (⚡ Static)';
        llmMatchingStatus.style.color = '#cbd5e0';
    }
}

// Load available templates
async function loadTemplates() {
    try {
        const response = await fetch(`${API_BASE}/api/templates`);
        const data = await response.json();
        
        if (data.success && data.templates) {
            templates = data.templates;
            displayTemplates(templates);
        }
    } catch (error) {
        console.error('Error loading templates:', error);
        templatesList.innerHTML = '<p style="color: #e53e3e;">Failed to load templates</p>';
    }
}

// Display templates
function displayTemplates(templates) {
    if (!templates || templates.length === 0) {
        templatesList.innerHTML = '<p style="color: #718096;">No templates available</p>';
        return;
    }
    
    let html = '';
    templates.forEach((template, idx) => {
        html += `
            <div class="template-item" onclick="useTemplate(${idx})">
                <div class="template-query">${template.user_query}</div>
                <div class="template-sql">${template.true_sql}</div>
            </div>
        `;
    });
    
    templatesList.innerHTML = html;
}

// Use a template
function useTemplate(index) {
    if (templates[index]) {
        queryInput.value = templates[index].user_query;
        queryInput.focus();
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Load databases
async function loadDatabases() {
    try {
        const response = await fetch(`${API_BASE}/api/databases`);
        const data = await response.json();
        
        if (data.success && data.databases) {
            databaseSelect.innerHTML = data.databases
                .map(db => `<option value="${db.value}">${db.label}</option>`)
                .join('');
            
            // Load catalogs for first database
            if (data.databases.length > 0) {
                await loadCatalogs(data.databases[0].value);
            }
        }
    } catch (error) {
        console.error('Error loading databases:', error);
        databaseSelect.innerHTML = '<option value="">Error loading databases</option>';
    }
}

// Load catalogs
async function loadCatalogs(database) {
    try {
        const response = await fetch(`${API_BASE}/api/catalogs?database=${database}`);
        const data = await response.json();
        
        if (data.success && data.catalogs) {
            allCatalogs = data.catalogs;
            catalogSelect.innerHTML = data.catalogs
                .map(cat => `<option value="${cat.value}">${cat.label}</option>`)
                .join('');
        }
    } catch (error) {
        console.error('Error loading catalogs:', error);
        catalogSelect.innerHTML = '<option value="">Error loading catalogs</option>';
    }
}

// Load models for provider
async function loadModelsForProvider(provider) {
    try {
        const response = await fetch(`${API_BASE}/api/models/${provider}`);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}: ${response.statusText}` }));
            throw new Error(errorData.detail || `Failed to load models: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.detail || 'Failed to load models');
        }
        
        if (data.models && data.models.length > 0) {
            modelSelect.innerHTML = data.models
                .map(model => `<option value="${model.value}">${model.label}</option>`)
                .join('');
            
            // For OpenRouter, select gpt-oss-20b if available (default), otherwise first model
            if (provider === 'openrouter') {
                const gptOss20bOption = Array.from(modelSelect.options).find(opt => 
                    opt.value.includes('gpt-oss-20b') || opt.value.includes('openai/gpt-oss-20b')
                );
                if (gptOss20bOption) {
                    modelSelect.value = gptOss20bOption.value;
                    console.log('Selected default OpenRouter model: GPT OSS 20B');
                } else if (data.models.length > 0) {
                    modelSelect.value = data.models[0].value;
                }
            } else if (data.models.length > 0) {
                // For other providers, select first model
                modelSelect.value = data.models[0].value;
            }
        } else {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            showAlert(`No ${provider} models found. Make sure the server is running and accessible.`, 'warning');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        const errorMessage = error.message || `Failed to load ${provider} models`;
        showAlert(errorMessage, 'error');
    }
}

// Load models for modal provider (for regeneration)
async function loadModelsForModalProvider(provider) {
    if (!modalModelSelect) return;
    
    try {
        modalModelSelect.innerHTML = '<option value="">Loading...</option>';
        
        const response = await fetch(`${API_BASE}/api/models/${provider}`);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}: ${response.statusText}` }));
            throw new Error(errorData.detail || `Failed to load models: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.detail || 'Failed to load models');
        }
        
        if (data.models && data.models.length > 0) {
            modalModelSelect.innerHTML = data.models
                .map(model => `<option value="${model.value}">${model.label}</option>`)
                .join('');
            
            // For OpenRouter, select gpt-oss-20b if available (default), otherwise first model
            if (provider === 'openrouter') {
                const gptOss20bOption = Array.from(modalModelSelect.options).find(opt => 
                    opt.value.includes('gpt-oss-20b') || opt.value.includes('openai/gpt-oss-20b')
                );
                if (gptOss20bOption) {
                    modalModelSelect.value = gptOss20bOption.value;
                } else if (data.models.length > 0) {
                    modalModelSelect.value = data.models[0].value;
                }
            } else if (data.models.length > 0) {
                // For other providers, select first model
                modalModelSelect.value = data.models[0].value;
            }
        } else {
            modalModelSelect.innerHTML = '<option value="">No models available</option>';
        }
    } catch (error) {
        console.error('Error loading models for modal:', error);
        modalModelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Show/hide loading
function showLoading(message) {
    loadingMessage.textContent = message;
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

// Show alert
function showAlert(message, type = 'error') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    document.body.appendChild(alert);
    
    setTimeout(() => alert.remove(), 5000);
}

// Handle Generate SQL
async function handleGenerateSQL() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showAlert('Please enter a question');
        return;
    }
    
    clearSQLResults();
    clearExecutionResults();
    
    const loadingMsg = autoExecute ? '⚡ Generating and executing SQL...' : '⚡ Generating SQL query...';
    showLoading(loadingMsg);
    
    try {
        const provider = providerSelect.value;
        const model = modelSelect.value;
        const target_db = databaseSelect.value;
        const catalog_db = databaseSelect.value;
        const catalog_name = catalogSelect.value;
        
        const response = await fetch(`${API_BASE}/api/smart-generate-sql`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query,
                model,
                request_limit: 5,
                iteration_limit: 5,
                temperature: 0.3,
                target_db,
                catalog_db,
                catalog_name,
                provider,
                use_templates: useTemplates,
                similarity_threshold: 0.6,
                auto_execute: autoExecute,
                use_llm_matching: useLLMMatching
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received response:', data);
        
        displaySQLResults(data);
        
        // Display execution results if auto-executed
        if (data.executed && data.execution) {
            console.log('Auto-execute was enabled, checking execution results...');
            if (data.execution.success) {
                console.log('Execution successful, displaying results...');
                displayExecutionResults(data.execution);
            } else {
                console.error('Execution failed:', data.execution.error);
                showAlert(`SQL executed but returned error: ${data.execution.error}`, 'warning');
            }
        } else {
            console.log('Not auto-executed or no execution data:', { 
                executed: data.executed, 
                hasExecution: !!data.execution 
            });
        }
        
    } catch (error) {
        console.error('Generation error:', error);
        showAlert(`SQL generation failed: ${error.message}`);
        
        // Show feedback UI even when generation fails
        // This allows users to manually enter SQL and provide feedback
        displayErrorWithFeedback(error.message);
    } finally {
        hideLoading();
    }
}

// Display error with feedback option
function displayErrorWithFeedback(errorMessage) {
    currentUserQuery = queryInput.value.trim();  // Store user query
    currentSQL = '';  // No SQL generated
    currentMethod = 'error';  // Mark as error
    
    // Show SQL results area with error message
    const sqlContent = document.getElementById('sqlContent');
    sqlContent.textContent = `❌ SQL Generation Failed: ${errorMessage}\n\nYou can manually enter SQL below and provide feedback to help improve the system.`;
    sqlContent.style.color = '#e53e3e';
    
    // Show error badge
    const badge = document.getElementById('performanceBadge');
    badge.innerHTML = `<span class="performance-badge badge-error">❌ Generation Failed</span>`;
    
    const templateInfo = document.getElementById('templateInfo');
    templateInfo.innerHTML = `<strong>⚠️ SQL generation failed</strong><br>Error: ${errorMessage}<br>You can manually enter the correct SQL and provide feedback.`;
    templateInfo.classList.remove('hidden');
    
    // Hide token usage
    document.getElementById('tokenUsage').textContent = '🎫 Tokens: 0 (Generation Failed)';
    
    // Show SQL results area
    sqlResults.classList.remove('hidden');
    
    // Show feedback section with manual SQL input option
    showFeedbackSectionForError();
    
    // Hide execute button (can't execute without SQL)
    executeBtn.style.display = 'none';
}

// Show feedback section when generation fails
function showFeedbackSectionForError() {
    const feedbackSection = document.getElementById('feedbackSection');
    if (feedbackSection) {
        feedbackSection.classList.remove('hidden');
        feedbackSection.style.display = 'block';
        
        // Update feedback section message
        const feedbackText = feedbackSection.querySelector('span');
        if (feedbackText) {
            feedbackText.textContent = 'SQL generation failed. Click "Dislike" to enter the correct SQL manually:';
        }
        
        // Show dislike button (to enter correct SQL)
        if (dislikeBtn) {
            dislikeBtn.style.display = 'inline-block';
            dislikeBtn.disabled = false;
        }
        // Hide like button (no SQL to like)
        if (likeBtn) {
            likeBtn.style.display = 'none';
        }
    }
}

// Display SQL Results
function displaySQLResults(data) {
    currentSQL = data.sql;
    currentUserQuery = queryInput.value.trim();  // Store user query
    currentMethod = data.method || 'pgai';  // Store generation method
    
    // Reset SQL content style
    const sqlContent = document.getElementById('sqlContent');
    sqlContent.textContent = data.sql;
    sqlContent.style.color = '';  // Reset color
    
    // Build time display
    let timeText = `⏱️ Generation: ${data.generation_time}s`;
    if (data.executed && data.execution_time !== undefined) {
        timeText += ` | Execution: ${data.execution_time}s`;
        const totalTime = parseFloat(data.generation_time) + parseFloat(data.execution_time);
        timeText += ` | Total: ${totalTime.toFixed(3)}s`;
    }
    document.getElementById('generationTime').textContent = timeText;
    
    // Show performance badge based on method
    const badge = document.getElementById('performanceBadge');
    const templateInfo = document.getElementById('templateInfo');
    
    // Handle different generation methods
    if (data.method === 'template-exact') {
        // Exact template match (100% similarity)
        badge.innerHTML = `<span class="performance-badge badge-template">📋 Template Match (Exact)</span>`;
        
        let infoHtml = `
            <strong>📋 Exact Template Match!</strong><br>
            Similarity: 100% | 
            Template: "${data.template_query}"<br>
            ${data.explanation || ''}
        `;
        
        if (data.executed && data.execution) {
            if (data.execution.success) {
                infoHtml += `<br>✅ Executed successfully - ${data.execution.row_count || 0} rows returned`;
            } else {
                infoHtml += `<br>❌ Execution failed`;
            }
        }
        
        templateInfo.innerHTML = infoHtml;
        templateInfo.classList.remove('hidden');
        
    } else if (data.method === 'template-llm' || data.method === 'llm-template') {
        // LLM-assisted template match with parameter binding
        badge.innerHTML = `<span class="performance-badge badge-template">🎯 Template Match (AI)</span>`;
        
        let infoHtml = `
            <strong>🎯 AI-Assisted Template Match!</strong><br>
            Similarity: ${(data.similarity_score * 100).toFixed(0)}% | 
            Template: "${data.template_query}"<br>
            ${data.explanation || 'Template-based SQL with AI parameter substitution'}
        `;
        
        if (data.executed && data.execution) {
            if (data.execution.success) {
                infoHtml += `<br>✅ Executed successfully - ${data.execution.row_count || 0} rows returned`;
            } else {
                infoHtml += `<br>❌ Execution failed`;
            }
        }
        
        templateInfo.innerHTML = infoHtml;
        templateInfo.classList.remove('hidden');
        
    } else if (data.method === 'template') {
        // Legacy template match (regex-based)
        badge.innerHTML = `<span class="performance-badge badge-template">⚡ Template Match</span>`;
        
        let infoHtml = `
            <strong>⚡ Template Match Found!</strong><br>
            Similarity: ${(data.similarity_score * 100).toFixed(0)}% | 
            Original Query: "${data.template_query}"<br>
            ${data.explanation || ''}
        `;
        
        if (data.executed && data.execution) {
            if (data.execution.success) {
                infoHtml += `<br>✅ Executed successfully - ${data.execution.row_count || 0} rows returned`;
            } else {
                infoHtml += `<br>❌ Execution failed`;
            }
        }
        
        templateInfo.innerHTML = infoHtml;
        templateInfo.classList.remove('hidden');
        
    } else {
        // PGAI - Generated from scratch
        badge.innerHTML = `<span class="performance-badge badge-pgai">🤖 AI Generated (Schema)</span>`;
        
        const templateInfo = document.getElementById('templateInfo');
        if (data.executed && data.execution) {
            let infoHtml = '';
            if (data.execution.success) {
                infoHtml = `<strong>✅ Executed successfully</strong> - ${data.execution.row_count || 0} rows returned`;
            } else {
                infoHtml = `<strong>❌ Execution failed</strong>`;
            }
            templateInfo.innerHTML = infoHtml;
            templateInfo.classList.remove('hidden');
        } else {
            templateInfo.classList.add('hidden');
        }
    }
    
    // Show token usage if available
    if (data.token_usage && data.token_usage.total_tokens > 0) {
        document.getElementById('tokenUsage').textContent = 
            `🎫 Tokens: ${data.token_usage.total_tokens} (in: ${data.token_usage.input_tokens}, out: ${data.token_usage.output_tokens})`;
    } else {
        document.getElementById('tokenUsage').textContent = '🎫 Tokens: 0 (Template Match)';
    }
    
    sqlResults.classList.remove('hidden');
    
    // Show feedback section for successful generation
    const feedbackSection = document.getElementById('feedbackSection');
    if (feedbackSection) {
        feedbackSection.classList.remove('hidden');
        feedbackSection.style.display = 'block';
        
        // Reset feedback section message
        const feedbackText = feedbackSection.querySelector('p');
        if (feedbackText) {
            feedbackText.textContent = 'Was this SQL helpful?';
        }
        
        // Show both like and dislike buttons
        if (likeBtn) {
            likeBtn.style.display = 'inline-block';
            likeBtn.disabled = false;
        }
        if (dislikeBtn) {
            dislikeBtn.style.display = 'inline-block';
            dislikeBtn.disabled = false;
        }
    }
    
    // Reset feedback status
    if (feedbackStatus) {
        feedbackStatus.textContent = '';
        feedbackStatus.style.display = 'none';
    }
    
    // Only show execute button if not auto-executed
    if (!data.executed) {
        executeBtn.style.display = 'inline-block';
    } else {
        executeBtn.style.display = 'inline-block'; // Show for re-execution
    }
}

// Handle Execute SQL
async function handleExecuteSQL() {
    if (!currentSQL) {
        showAlert('No SQL to execute');
        return;
    }
    
    showLoading('▶️ Executing query...');
    
    try {
        const target_db = databaseSelect.value;
        
        const response = await fetch(`${API_BASE}/api/execute-sql`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                sql: currentSQL,
                target_db
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayExecutionResults(data);
        
    } catch (error) {
        console.error('Execution error:', error);
        showAlert(`Query execution failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display Execution Results
function displayExecutionResults(data) {
    console.log('Displaying execution results:', data);
    const executionContent = document.getElementById('executionContent');
    
    if (!executionContent) {
        console.error('executionContent element not found!');
        return;
    }
    
    if (data.rows && data.rows.length > 0) {
        console.log(`Rendering table with ${data.row_count} rows and ${data.columns.length} columns`);
        let html = `<p style="color: var(--success-color); margin-bottom: 1rem;">✅ <strong>${data.row_count} rows returned</strong></p>`;
        
        // Create table
        html += '<div style="overflow-x: auto; max-height: 600px;"><table class="result-table"><thead><tr>';
        
        // Headers
        data.columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Rows (limit to 100 for display)
        const displayRows = data.rows.slice(0, 100);
        displayRows.forEach(row => {
            html += '<tr>';
            data.columns.forEach(col => {
                const value = row[col];
                const displayValue = value !== null && value !== undefined ? value : '<i>null</i>';
                html += `<td>${displayValue}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table></div>';
        
        if (data.rows.length > 100) {
            html += `<p style="margin-top: 10px; color: #94a3b8;">Showing first 100 of ${data.row_count} rows</p>`;
        }
        
        executionContent.innerHTML = html;
        console.log('Table HTML rendered successfully');
    } else {
        executionContent.innerHTML = '<p style="color: var(--success-color);">✅ Query executed successfully (no rows returned)</p>';
    }
    
    executionResults.classList.remove('hidden');
    console.log('executionResults section made visible');
    
    // Scroll to results
    executionResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle Copy SQL
function handleCopySQL() {
    if (!currentSQL) {
        showAlert('No SQL to copy');
        return;
    }
    
    navigator.clipboard.writeText(currentSQL).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        showAlert('Failed to copy SQL');
    });
}

// Clear results
function clearSQLResults() {
    sqlResults.classList.add('hidden');
    currentSQL = '';
    currentUserQuery = '';
    currentMethod = '';
    executeBtn.style.display = 'none';
    
    // Reset feedback buttons
    if (likeBtn) {
        likeBtn.disabled = false;
    }
    if (dislikeBtn) {
        dislikeBtn.disabled = false;
    }
    if (feedbackStatus) {
        feedbackStatus.textContent = '';
        feedbackStatus.style.display = 'none';
    }
}

function clearExecutionResults() {
    executionResults.classList.add('hidden');
}

// Feedback Handlers
async function handleLikeFeedback() {
    if (!currentSQL || !currentUserQuery) {
        showAlert('No SQL to provide feedback on');
        return;
    }
    
    try {
        // Disable buttons during processing
        likeBtn.disabled = true;
        dislikeBtn.disabled = true;
        feedbackStatus.textContent = '⏳ Processing feedback...';
        feedbackStatus.style.display = 'block';
        feedbackStatus.style.color = '#667eea';
        
        const provider = providerSelect.value;
        const model = modelSelect.value;
        
        if (!model || !model.trim()) {
            throw new Error('Please select a model');
        }
        if (!provider || !provider.trim()) {
            throw new Error('Please select a provider');
        }
        
        const response = await fetch(`${API_BASE}/api/feedback/like`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_query: currentUserQuery,
                true_sql: currentSQL,
                model: model,
                provider: provider,
                feedback: 'like'
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            feedbackStatus.textContent = '✅ Feedback submitted! Template added successfully.';
            feedbackStatus.style.color = '#48bb78';
            
            // Reload templates list
            await loadTemplates();
            
            // Disable feedback buttons after successful submission
            likeBtn.disabled = true;
            dislikeBtn.disabled = true;
        } else {
            throw new Error(data.message || 'Failed to submit feedback');
        }
    } catch (error) {
        console.error('Feedback error:', error);
        feedbackStatus.textContent = `❌ Error: ${error.message}`;
        feedbackStatus.style.color = '#f56565';
        likeBtn.disabled = false;
        dislikeBtn.disabled = false;
    }
}

async function handleDislikeFeedback() {
    if (!currentUserQuery) {
        showAlert('No query to provide feedback on');
        return;
    }
    
    // Show modal (even if SQL generation failed)
    if (dislikeModal) {
        if (modalUserQuery) {
            modalUserQuery.textContent = currentUserQuery;
        }
        if (modalGeneratedSql) {
            modalGeneratedSql.textContent = currentSQL || 'SQL generation failed - please enter correct SQL manually';
        }
        if (correctSqlInput) {
            correctSqlInput.value = currentSQL || '';  // Pre-fill with generated SQL if available, empty if error
            // Focus input if empty (error case)
            if (!currentSQL) {
                setTimeout(() => correctSqlInput.focus(), 100);
            }
        }
        
        // Initialize modal provider/model dropdowns
        // Default to OpenRouter for regeneration (stronger model)
        if (modalProviderSelect) {
            modalProviderSelect.value = 'openrouter';
            // Load models for the selected provider
            await loadModelsForModalProvider('openrouter');
        }
        
        dislikeModal.classList.remove('hidden');
        dislikeModal.style.display = 'block';
    }
}

function closeDislikeModal() {
    if (dislikeModal) {
        dislikeModal.classList.add('hidden');
        dislikeModal.style.display = 'none';
        correctSqlInput.value = '';
    }
}

async function handleRegenerateSQL() {
    if (!currentUserQuery) {
        showAlert('No query to regenerate');
        return;
    }
    
    // Get provider and model from modal dropdowns
    const provider = modalProviderSelect ? modalProviderSelect.value : providerSelect.value;
    const model = modalModelSelect ? modalModelSelect.value : modelSelect.value;
    
    if (!model || !model.trim()) {
        showAlert('Please select a model for regeneration');
        return;
    }
    
    if (!provider || !provider.trim()) {
        showAlert('Please select a provider for regeneration');
        return;
    }
    
    try {
        regenerateSqlBtn.disabled = true;
        regenerateSqlBtn.textContent = '⏳ Regenerating...';
        
        // Determine base URL
        let base_url = null;
        if (provider === 'openrouter') {
            base_url = 'https://openrouter.ai/api/v1';
        } else if (provider === 'openai') {
            base_url = 'https://api.openai.com/v1';
        } else if (provider === 'ollama') {
            base_url = null; // Will use default from backend
        }
        
        const response = await fetch(`${API_BASE}/api/feedback/regenerate-sql`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_query: currentUserQuery,
                model: model,
                provider: provider,
                base_url: base_url
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success && data.sql) {
            correctSqlInput.value = data.sql;
            showAlert('SQL regenerated successfully', 'success');
        } else {
            throw new Error(data.message || 'Failed to regenerate SQL');
        }
    } catch (error) {
        console.error('Regeneration error:', error);
        showAlert(`Failed to regenerate SQL: ${error.message}`, 'error');
    } finally {
        regenerateSqlBtn.disabled = false;
        regenerateSqlBtn.textContent = '🤖 Regenerate with AI';
    }
}

async function handleSubmitFeedback() {
    const correctSql = correctSqlInput.value.trim();
    
    if (!correctSql) {
        showAlert('Please provide the correct SQL');
        return;
    }
    
    if (!currentUserQuery) {
        showAlert('No query associated with this feedback');
        return;
    }
    
    try {
        submitFeedbackBtn.disabled = true;
        submitFeedbackBtn.textContent = '⏳ Submitting...';
        
        const provider = providerSelect.value;
        const model = modelSelect.value;
        
        const response = await fetch(`${API_BASE}/api/feedback/like`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_query: currentUserQuery,
                true_sql: correctSql,
                model: model,
                provider: provider,
                feedback: 'dislike'  // Mark as dislike since it required correction
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('✅ Feedback submitted! Template added with corrected SQL.', 'success');
            
            // Close modal
            closeDislikeModal();
            
            // Reload templates list
            await loadTemplates();
            
            // Update feedback status
            if (feedbackStatus) {
                feedbackStatus.textContent = '✅ Feedback submitted! Template added with corrected SQL.';
                feedbackStatus.style.color = '#48bb78';
                feedbackStatus.style.display = 'block';
            }
            
            // Disable feedback buttons
            likeBtn.disabled = true;
            dislikeBtn.disabled = true;
        } else {
            throw new Error(data.message || 'Failed to submit feedback');
        }
    } catch (error) {
        console.error('Feedback submission error:', error);
        showAlert(`Failed to submit feedback: ${error.message}`, 'error');
    } finally {
        submitFeedbackBtn.disabled = false;
        submitFeedbackBtn.textContent = '✅ Submit Feedback';
    }
}

