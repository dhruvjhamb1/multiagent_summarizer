const API_BASE = window.location.origin;
let currentDocumentId = null, currentJobId = null, statusPollingInterval = null;

const formatBytes = bytes => bytes === 0 ? '0 Bytes' : `${(bytes / Math.pow(1024, Math.floor(Math.log(bytes) / Math.log(1024)))).toFixed(2)} ${['Bytes', 'KB', 'MB', 'GB'][Math.floor(Math.log(bytes) / Math.log(1024))]}`;

const showNotification = (message, type = 'success') => {
    const notification = document.getElementById('notification');
    const colors = { success: 'bg-green-50 text-green-800 border border-green-200', error: 'bg-red-50 text-red-800 border border-red-200', info: 'bg-blue-50 text-blue-800 border border-blue-200' };
    notification.className = `fixed top-4 right-4 max-w-md p-4 rounded-lg shadow-2xl z-50 ${colors[type]}`;
    document.getElementById('notification-message').textContent = message;
    notification.classList.remove('hidden');
    setTimeout(() => notification.classList.add('hidden'), 5000);
};

// File Upload
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.onclick = () => fileInput.click();
uploadZone.ondragover = e => { e.preventDefault(); uploadZone.classList.add('drag-over'); };
uploadZone.ondragleave = () => uploadZone.classList.remove('drag-over');
uploadZone.ondrop = e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
};

fileInput.onchange = e => e.target.files.length && handleFileSelect(e.target.files[0]);

const handleFileSelect = file => {
    if (!['application/pdf', 'text/plain'].includes(file.type)) return showNotification('Invalid file type. Use PDF or TXT.', 'error');
    if (file.size > 10 * 1024 * 1024) return showNotification('File too large. Max 10MB.', 'error');
    
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatBytes(file.size);
    document.getElementById('file-info').classList.remove('hidden');
    document.getElementById('upload-btn').disabled = false;
};

document.getElementById('remove-file').onclick = e => {
    e.stopPropagation();
    fileInput.value = '';
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('upload-btn').disabled = true;
};

document.getElementById('upload-btn').onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
        
        const data = await res.json();
        currentDocumentId = data.document_id;
        
        showNotification('Document uploaded successfully!', 'success');
        document.getElementById('document-id').textContent = currentDocumentId;
        document.getElementById('analysis-section').classList.remove('hidden');
        
        fileInput.value = '';
        document.getElementById('file-info').classList.add('hidden');
        document.getElementById('upload-btn').disabled = true;
    } catch (error) {
        showNotification(error.message, 'error');
    }
};

// Analysis
document.getElementById('analyze-btn').onclick = async () => {
    if (!currentDocumentId) return;
    
    try {
        const res = await fetch(`${API_BASE}/analyze/${currentDocumentId}`, { method: 'POST' });
        if (!res.ok) throw new Error((await res.json()).detail || 'Analysis failed');
        
        const data = await res.json();
        currentJobId = data.job_id;
        
        showNotification('Analysis started!', 'success');
        document.getElementById('job-id').textContent = currentJobId;
        document.getElementById('status-section').classList.remove('hidden');
        
        startStatusPolling();
    } catch (error) {
        showNotification(error.message, 'error');
    }
};

// Status Polling
const startStatusPolling = () => {
    if (statusPollingInterval) clearInterval(statusPollingInterval);
    pollStatus();
    statusPollingInterval = setInterval(pollStatus, 2000);
};

const pollStatus = async () => {
    if (!currentJobId) return;
    
    try {
        const res = await fetch(`${API_BASE}/status/${currentJobId}`);
        const data = await res.json();
        
        const statusColors = { pending: 'bg-yellow-100 text-yellow-800', processing: 'bg-blue-100 text-blue-800', completed: 'bg-green-100 text-green-800', failed: 'bg-red-100 text-red-800', partial: 'bg-orange-100 text-orange-800' };
        document.getElementById('job-status').textContent = data.status.toUpperCase();
        document.getElementById('job-status').className = `px-3 py-1 rounded text-xs font-bold uppercase ${statusColors[data.status]}`;
        document.getElementById('progress-bar').style.width = `${data.progress_percentage || 0}%`;
        document.getElementById('progress-percent').textContent = `${Math.round(data.progress_percentage || 0)}%`;
        
        const agentStatus = document.getElementById('agent-status');
        agentStatus.innerHTML = ['summarizer', 'entity_extractor', 'sentiment_analyzer'].map(agent => {
            const status = data.agents_status?.[agent] || 'pending';
            const colors = { completed: 'bg-green-500', processing: 'bg-blue-500', pending: 'bg-yellow-500', failed: 'bg-red-500' };
            return `<div class="flex justify-between"><span><span class="inline-block w-3 h-3 rounded-full mr-2 ${colors[status]}"></span>${agent.replace('_', ' ')}</span><span class="text-xs font-semibold">${status.toUpperCase()}</span></div>`;
        }).join('');
        
        if (['completed', 'failed', 'partial'].includes(data.status)) {
            clearInterval(statusPollingInterval);
            fetchResults();
        }
    } catch (error) {
        console.error('Status error:', error);
    }
};

// Results
const fetchResults = async () => {
    if (!currentJobId) return;
    
    try {
        const res = await fetch(`${API_BASE}/results/${currentJobId}`);
        const data = await res.json();
        displayResults(data);
    } catch (error) {
        showNotification('Failed to fetch results', 'error');
    }
};

const displayResults = data => {
    document.getElementById('results-section').classList.remove('hidden');
    const content = document.getElementById('results-content');
    content.innerHTML = '';
    
    if (data.status === 'processing' || data.status === 'pending') {
        content.innerHTML = '<div class="text-center py-8"><div class="spinner w-12 h-12 rounded-full mx-auto"></div><p class="text-gray-600 mt-4">Processing...</p></div>';
        return;
    }
    
    // Metadata
    if (data.metadata) {
        const m = data.metadata;
        let failedAgentsHtml = '';
        
        if (m.failed_agents && m.failed_agents.length > 0) {
            failedAgentsHtml = `
                <div class="col-span-2 mt-2 p-3 bg-red-50 border border-red-200 rounded">
                    <h4 class="font-semibold text-sm text-red-800 mb-2">Failed Agents:</h4>
                    <div class="space-y-1">
                        ${m.failed_agents.map(agent => {
                            // Get error message from results if available
                            const errorMsg = data.results?.[agent]?.error || 'Unknown error';
                            return `<div class="text-xs text-red-700">• <span class="font-semibold">${agent.replace('_', ' ')}</span>: ${errorMsg}</div>`;
                        }).join('')}
                    </div>
                </div>
            `;
        }
        
        content.innerHTML += `
            <div class="bg-purple-50 rounded-xl p-4 border border-purple-200">
                <h3 class="font-bold mb-3">Metadata</h3>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div>
                        <span class="text-gray-600">Processing Time:</span>
                        <span class="font-semibold text-gray-800 ml-2">${m.total_processing_time_seconds?.toFixed(2)}s</span>
                    </div>
                    <div>
                        <span class="text-gray-600">Parallel Execution:</span>
                        <span class="font-semibold text-gray-800 ml-2">${m.parallel_execution ? 'Yes' : 'No'}</span>
                    </div>
                    <div>
                        <span class="text-gray-600">Agents Completed:</span>
                        <span class="font-semibold text-green-600 ml-2">${m.agents_completed}</span>
                    </div>
                    <div>
                        <span class="text-gray-600">Agents Failed:</span>
                        <span class="font-semibold text-red-600 ml-2">${m.agents_failed}</span>
                    </div>
                    ${m.timestamp ? `
                        <div class="col-span-2">
                            <span class="text-gray-600">Timestamp:</span>
                            <span class="font-semibold text-gray-800 ml-2">${new Date(m.timestamp).toLocaleString()}</span>
                        </div>
                    ` : ''}
                    ${failedAgentsHtml}
                </div>
                ${m.warning ? `<p class="text-yellow-800 text-sm mt-3 p-2 bg-yellow-50 border border-yellow-200 rounded">⚠️ ${m.warning}</p>` : ''}
            </div>
        `;
    }
    
    // Summary
    if (data.results?.summary && !data.results.summary.error) {
        const s = data.results.summary;
        content.innerHTML += `<div class="bg-white rounded-xl p-4 border"><h3 class="font-bold mb-2">📄 Summary</h3><p class="text-gray-700 text-sm">${s.text}</p>${s.key_points?.length ? `<h5 class="font-semibold mt-4 mb-2">Key Points:</h5><ul class="list-disc list-inside mt-2 text-sm">${s.key_points.map(p => `<li>${p}</li>`).join('')}</ul>` : ''}<div class="flex justify-between text-xs text-gray-500 mt-3"><span>Confidence: ${(s.confidence * 100).toFixed(1)}%</span><span>Processing Time: ${s.processing_time?.toFixed(2)}s</span></div></div>`;
    }
    
    // Entities
    if (data.results?.entities && !data.results.entities.error) {
        const e = data.results.entities;
        const formatEntity = entity => {
            if (typeof entity === 'string') return entity;
            if (entity.name && entity.role) return `${entity.name} (${entity.role})${entity.mentions ? ` - ${entity.mentions} mention${entity.mentions > 1 ? 's' : ''}` : ''}`;
            if (entity.name && entity.type) return `${entity.name} (${entity.type})${entity.mentions ? ` - ${entity.mentions} mention${entity.mentions > 1 ? 's' : ''}` : ''}`;
            if (entity.name) return `${entity.name}${entity.type ? ` (${entity.type})` : ''}${entity.mentions ? ` - ${entity.mentions} mention${entity.mentions > 1 ? 's' : ''}` : ''}`;
            if (entity.date) return `${entity.date}${entity.context ? ` - ${entity.context}` : ''}`;
            if (entity.amount) return `${entity.amount}${entity.context ? ` - ${entity.context}` : ''}`;
            return JSON.stringify(entity);
        };
        
        let entitiesHtml = '';
        [['people', '👤'], ['organizations', '🏢'], ['locations', '📍'], ['dates', '📅'], ['monetary_values', '💰']].forEach(([key, icon]) => {
            if (e[key]?.length) {
                entitiesHtml += `<div class="mb-3"><h4 class="font-semibold text-sm mb-2">${icon} ${key.replace('_', ' ')}</h4><div class="space-y-1">${e[key].map(entity => `<div class="px-3 py-1 bg-green-50 text-green-800 rounded text-xs border border-green-200">${formatEntity(entity)}</div>`).join('')}</div></div>`;
            }
        });
        
        content.innerHTML += `<div class="bg-white rounded-xl p-4 border"><h3 class="font-bold mb-2">🏷️ Entities</h3>${entitiesHtml || '<p class="text-sm text-gray-500">No entities found</p>'}<p class="text-xs text-gray-500 mt-3">Processing Time: ${e.processing_time?.toFixed(2)}s</p></div>`;
    }
    
    // Sentiment
    if (data.results?.sentiment && !data.results.sentiment.error) {
        const s = data.results.sentiment;
        const colors = { positive: 'bg-green-100 text-green-800', negative: 'bg-red-100 text-red-800', neutral: 'bg-gray-100 text-gray-800' };
        
        let emotionalHtml = '';
        if (s.emotional_indicators && Object.keys(s.emotional_indicators).length > 0) {
            emotionalHtml = `<div class="mt-3"><h4 class="font-semibold text-sm mb-2">Emotional Indicators</h4><div class="flex flex-wrap gap-2">${Object.entries(s.emotional_indicators).map(([key, val]) => `<span class="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">${key}: ${typeof val === 'number' ? val.toFixed(2) : val}</span>`).join('')}</div></div>`;
        }
        
        let phrasesHtml = '';
        if (s.key_phrases && s.key_phrases.length > 0) {
            phrasesHtml = `<div class="mt-3"><h4 class="font-semibold text-sm mb-2">Key Phrases</h4><div class="space-y-2">${s.key_phrases.map(p => `<div class="flex justify-between items-start p-2 bg-gray-50 rounded text-xs"><span class="text-gray-700">"${p.text}"</span><span class="px-2 py-1 rounded font-semibold ${colors[p.sentiment] || colors.neutral}">${p.sentiment}</span></div>`).join('')}</div></div>`;
        }
        
        content.innerHTML += `<div class="bg-white rounded-xl p-4 border"><h3 class="font-bold mb-2">😊 Sentiment</h3><div class="flex justify-between items-center mb-3"><span class="px-4 py-2 rounded font-bold ${colors[s.overall] || colors.neutral}">${s.overall?.toUpperCase()}</span><span class="text-2xl font-bold">${(s.confidence * 100).toFixed(1)}%</span></div>${s.tone ? `<div class="grid grid-cols-3 gap-2 text-xs mb-3"><div class="bg-gray-50 p-2 rounded text-center"><p class="text-gray-600">Formality</p><p class="font-semibold">${s.tone.formality}</p></div><div class="bg-gray-50 p-2 rounded text-center"><p class="text-gray-600">Urgency</p><p class="font-semibold">${s.tone.urgency}</p></div><div class="bg-gray-50 p-2 rounded text-center"><p class="text-gray-600">Objectivity</p><p class="font-semibold">${s.tone.objectivity}</p></div></div>` : ''}${emotionalHtml}${phrasesHtml}<p class="text-xs text-gray-500 mt-3">Processing Time: ${s.processing_time?.toFixed(2)}s</p></div>`;
    }
    
    showNotification(data.status === 'completed' ? 'Analysis completed!' : data.status === 'partial' ? 'Partial results available' : 'Analysis failed', data.status === 'completed' ? 'success' : 'info');
};
