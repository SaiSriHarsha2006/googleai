document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------
    // Tab Switching
    // -----------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    const tabMeta = {
        'tab-generator': {
            title: 'AI Code Generator',
            subtitle: 'Generate code snippets using pretrained LLMs or custom trained transformers.'
        },
        'tab-chatbot': {
            title: 'Gemini Coding Assistant',
            subtitle: 'Converse with Google Gemini to write code, design algorithms, or debug programming queries.'
        },
        'tab-train': {
            title: 'Train Scratch Model',
            subtitle: 'Train a custom character-level Transformer on Python code and observe loss optimization live.'
        },
        'tab-explain': {
            title: 'Under the Hood',
            subtitle: 'Learn the architectural principles behind generative language models and code builders.'
        }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Toggle active classes
            navButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(t => t.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            
            // Update Header Meta
            pageTitle.textContent = tabMeta[tabId].title;
            pageSubtitle.textContent = tabMeta[tabId].subtitle;
        });
    });

    // -----------------------------------------
    // Copy Code Button
    // -----------------------------------------
    const copyBtn = document.getElementById('copy-btn');
    const outputBlock = document.getElementById('output-block');

    copyBtn.addEventListener('click', () => {
        const text = outputBlock.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.style.borderColor = 'var(--accent-green)';
            copyBtn.style.color = 'var(--accent-green)';
            
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.style.borderColor = '';
                copyBtn.style.color = '';
            }, 2000);
        });
    });

    // -----------------------------------------
    // Code Generator
    // -----------------------------------------
    const generateBtn = document.getElementById('generate-btn');
    const modelSelect = document.getElementById('model-select');
    const promptInput = document.getElementById('prompt-input');
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const tokensInput = document.getElementById('tokens-input');

    // Sync temperature text
    tempSlider.addEventListener('input', (e) => {
        tempVal.textContent = e.target.value;
    });

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt first.');
            return;
        }

        const model = modelSelect.value;
        const temperature = parseFloat(tempSlider.value);
        const maxTokens = parseInt(tokensInput.value, 10);

        // Toggle Loading State
        generateBtn.disabled = true;
        generateBtn.querySelector('.loader-spinner').classList.remove('hidden');
        outputBlock.textContent = '# Model thinking... (Please wait, this might take a moment to compute/load model)';

        const endpoint = model === 'pretrained' ? '/api/generate/pretrained' : '/api/generate/scratch';
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    temperature: temperature,
                    max_tokens: maxTokens
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                // Display code with typewriter animation
                typewriterEffect(outputBlock, data.code);
            } else {
                outputBlock.textContent = `# Error: ${data.message || 'Failed to generate code.'}`;
            }
        } catch (error) {
            outputBlock.textContent = `# Network Error: Could not reach generator backend.\nDetails: ${error.message}`;
        } finally {
            generateBtn.disabled = false;
            generateBtn.querySelector('.loader-spinner').classList.add('hidden');
        }
    });

    function typewriterEffect(element, text) {
        element.textContent = '';
        let i = 0;
        const speed = text.length > 300 ? 5 : 15; // Speed up for long outputs
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    // -----------------------------------------
    // Training Dashboard & Loss Chart
    // -----------------------------------------
    const startTrainBtn = document.getElementById('start-train-btn');
    const trainEpochsInput = document.getElementById('train-epochs');
    const trainLrInput = document.getElementById('train-lr');
    const trainBatchSelect = document.getElementById('train-batch');
    
    const trainStatus = document.getElementById('train-status');
    const trainEpochVal = document.getElementById('train-epoch-val');
    const trainLossVal = document.getElementById('train-loss-val');
    const valLossVal = document.getElementById('val-loss-val');
    const trainProgressBar = document.getElementById('train-progress-bar');
    const consoleStream = document.getElementById('console-stream');

    let ws = null;
    let lossChart = null;

    // Initialize Chart.js
    function initChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        if (lossChart) {
            lossChart.destroy();
        }

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.05)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.05)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Inter' } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748b' }
                    }
                }
            }
        });
    }

    // Run chart init once
    initChart();

    function writeConsole(text, type = 'system-line') {
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        line.textContent = text;
        consoleStream.appendChild(line);
        consoleStream.scrollTop = consoleStream.scrollHeight;
    }

    startTrainBtn.addEventListener('click', () => {
        // If already running, prevent clicks
        if (ws && ws.readyState === WebSocket.OPEN) return;

        const epochs = parseInt(trainEpochsInput.value, 10);
        const lr = parseFloat(trainLrInput.value);
        const batchSize = parseInt(trainBatchSelect.value, 10);

        // Reset elements
        initChart();
        trainProgressBar.style.width = '0%';
        consoleStream.innerHTML = '';
        writeConsole('[System] Initiating WebSocket connection to training backend...', 'system-line');

        // Set UI state
        startTrainBtn.disabled = true;
        startTrainBtn.textContent = 'Training in Progress...';
        trainStatus.textContent = 'Running';
        trainStatus.className = 'val status-running';

        // Connect WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/train`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            writeConsole('[System] WebSocket opened. Sending training configuration...', 'system-line');
            ws.send(JSON.stringify({
                epochs: epochs,
                lr: lr,
                batch_size: batchSize
            }));
            writeConsole('[System] Training started. Awaiting PyTorch updates...', 'system-line');
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            if (msg.type === 'progress') {
                const pct = ((msg.epoch / epochs) * 100).toFixed(1);
                trainProgressBar.style.width = `${pct}%`;
                
                // Update UI Labels
                trainEpochVal.textContent = `${msg.epoch} / ${epochs}`;
                trainLossVal.textContent = msg.train_loss.toFixed(4);
                valLossVal.textContent = msg.val_loss.toFixed(4);
                
                // Print progress to terminal console
                writeConsole(`[Epoch ${msg.epoch}/${epochs}] Train Loss: ${msg.train_loss.toFixed(4)} | Val Loss: ${msg.val_loss.toFixed(4)}`, 'progress-line');
                
                // Print code sample
                writeConsole(`[Generated Sample Code]:\n${msg.sample}`, 'sample-line');

                // Update Chart
                lossChart.data.labels.push(msg.epoch);
                lossChart.data.datasets[0].data.push(msg.train_loss);
                lossChart.data.datasets[1].data.push(msg.val_loss);
                lossChart.update();

            } else if (msg.type === 'complete') {
                trainProgressBar.style.width = '100%';
                trainStatus.textContent = 'Completed';
                trainStatus.className = 'val status-completed';
                writeConsole('[System] Training completed successfully! Custom model weights and meta token keys saved to data/ directory.', 'success-line');
                resetTrainBtn();

            } else if (msg.type === 'error') {
                trainStatus.textContent = 'Error';
                trainStatus.className = 'val status-error';
                writeConsole(`[Error] Training aborted: ${msg.message}`, 'error-line');
                resetTrainBtn();
            }
        };

        ws.onerror = (err) => {
            writeConsole('[Error] WebSocket error occurred.', 'error-line');
            resetTrainBtn();
        };

        ws.onclose = () => {
            writeConsole('[System] WebSocket connection closed.', 'system-line');
            resetTrainBtn();
        };
    });

    function resetTrainBtn() {
        startTrainBtn.disabled = false;
        startTrainBtn.textContent = 'Start Model Training';
        if (ws) {
            ws.close();
            ws = null;
        }
    }

    // -----------------------------------------
    // Gemini Chatbot Controller
    // -----------------------------------------
    const chatSettingsBtn = document.getElementById('chat-settings-btn');
    const chatApiKeyPanel = document.getElementById('chat-api-key-panel');
    const chatApiKeyInput = document.getElementById('chat-api-key-input');
    const saveApiKeyBtn = document.getElementById('save-api-key-btn');
    const chatApiStatus = document.getElementById('chat-api-status');
    const chatMessagesContainer = document.getElementById('chat-messages-container');
    const chatInputTextarea = document.getElementById('chat-input-textarea');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatThinkingIndicator = document.getElementById('chat-thinking-indicator');
    const chatModelSelect = document.getElementById('chat-model-select');

    let chatHistory = []; // Holds message history in frontend

    // Load key from localStorage on start
    let savedApiKey = localStorage.getItem('gemini_api_key') || '';
    if (savedApiKey) {
        chatApiKeyInput.value = savedApiKey;
        updateApiStatus(true);
    } else {
        updateApiStatus(false);
    }

    function updateApiStatus(hasKey) {
        if (hasKey) {
            chatApiStatus.textContent = 'Active (Stored)';
            chatApiStatus.className = 'chat-status status-active';
        } else {
            chatApiStatus.textContent = 'Key Missing (Using Env)';
            chatApiStatus.className = 'chat-status status-key-missing';
        }
    }

    // Toggle API Key settings panel
    chatSettingsBtn.addEventListener('click', () => {
        chatApiKeyPanel.classList.toggle('hidden');
    });

    // Save API key
    saveApiKeyBtn.addEventListener('click', () => {
        const key = chatApiKeyInput.value.trim();
        if (key) {
            localStorage.setItem('gemini_api_key', key);
            savedApiKey = key;
            updateApiStatus(true);
            alert('API Key saved successfully!');
            chatApiKeyPanel.classList.add('hidden');
        } else {
            localStorage.removeItem('gemini_api_key');
            savedApiKey = '';
            updateApiStatus(false);
            alert('API Key removed. System will fall back to backend environmental variable.');
        }
    });

    // Handle sending message
    async function handleSendMessage() {
        const message = chatInputTextarea.value.trim();
        if (!message) return;

        // Clear input area
        chatInputTextarea.value = '';
        chatInputTextarea.style.height = 'auto'; // Reset textarea height

        // Render User Message
        appendMessage('user', message);
        
        // Show thinking bubble
        chatThinkingIndicator.classList.remove('hidden');
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

        try {
            // Call chatbot API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory,
                    apiKey: savedApiKey,
                    modelName: chatModelSelect.value
                })
            });

            const data = await response.json();
            
            // Hide thinking indicator
            chatThinkingIndicator.classList.add('hidden');

            if (data.status === 'success') {
                // Render assistant message
                appendMessage('model', data.reply);
                
                // Add to history
                chatHistory.push({ role: 'user', text: message });
                chatHistory.push({ role: 'model', text: data.reply });
            } else {
                appendMessage('assistant-error', `Error: ${data.message}`);
            }
        } catch (error) {
            chatThinkingIndicator.classList.add('hidden');
            appendMessage('assistant-error', `Network Error: Failed to contact backend.\nDetails: ${error.message}`);
        }

        // Scroll to bottom
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // Append a message bubble to chat window
    function appendMessage(role, text) {
        const messageDiv = document.createElement('div');
        
        if (role === 'user') {
            messageDiv.className = 'chat-message user-message';
        } else if (role === 'model') {
            messageDiv.className = 'chat-message assistant-message';
        } else {
            messageDiv.className = 'chat-message assistant-message error-message';
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (role === 'model') {
            contentDiv.innerHTML = formatMarkdown(text);
        } else {
            // Escape html to prevent XSS
            const p = document.createElement('p');
            p.textContent = text;
            contentDiv.appendChild(p);
        }

        messageDiv.appendChild(contentDiv);
        chatMessagesContainer.appendChild(messageDiv);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // Format simple Markdown: replaces code blocks and bold text
    function formatMarkdown(text) {
        // Regex to match code blocks ```language ... ```
        const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
        let formatted = text;
        
        formatted = formatted.replace(codeBlockRegex, (match, lang, code) => {
            const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<pre><code class="language-${lang || 'plaintext'}">${escapedCode}</code></pre>`;
        });
        
        // Inline code blocks `code`
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Paragraph formatting: splits text into lines and wraps paragraphs
        const lines = formatted.split('\n\n');
        return lines.map(line => {
            if (line.startsWith('<pre>')) return line;
            return `<p>${line.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    }

    // Event listeners for sending
    chatSendBtn.addEventListener('click', handleSendMessage);
    
    chatInputTextarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-grow textarea height slightly for long messages
    chatInputTextarea.addEventListener('input', () => {
        chatInputTextarea.style.height = 'auto';
        chatInputTextarea.style.height = (chatInputTextarea.scrollHeight) + 'px';
    });
});
