/**
 * Elliott Wave Bot - Frontend JavaScript
 * Handles user interactions and API communication
 * Enhanced with ML Features: Wave Accuracy, Auto-Tuning, Backtesting
 */

let currentAnalysis = null;
let isAnalyzing = false;
let mlFeaturesAvailable = false;

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🚀 Elliott Wave Bot - Initializing...');
    
    // Check ML features availability
    checkMLFeatures();
    
    // Initialize event listeners
    setupEventListeners();
    
    // Populate trading pairs for default category (forex)
    updateTradingPairs('forex');
    
    console.log('✅ Elliott Wave Bot - Ready!');
}

/**
 * Check if ML features are available
 */
function checkMLFeatures() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            mlFeaturesAvailable = data.ml_features || false;
            updateMLButtonsVisibility();
            console.log(`🤖 ML Features: ${mlFeaturesAvailable ? 'Available' : 'Not Available'}`);
        })
        .catch(error => {
            console.log('⚠️ Could not check ML features availability');
            mlFeaturesAvailable = false;
            updateMLButtonsVisibility();
        });
}

/**
 * Update ML buttons visibility based on availability
 */
function updateMLButtonsVisibility() {
    const mlButtons = document.querySelectorAll('.ml-feature-btn');
    mlButtons.forEach(btn => {
        if (mlFeaturesAvailable) {
            btn.style.display = 'inline-block';
            btn.disabled = false;
        } else {
            btn.style.display = 'none';
            btn.disabled = true;
        }
    });
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Category selection
    document.querySelectorAll('input[name="category"]').forEach(radio => {
        radio.addEventListener('change', function() {
            updateTradingPairs(this.value);
        });
    });
    
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', performAnalysis);
    
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', performAnalysis);
    
    // Export button
    document.getElementById('exportBtn').addEventListener('click', exportChart);
    
    // ML Feature buttons
    if (document.getElementById('mlAccuracyBtn')) {
        document.getElementById('mlAccuracyBtn').addEventListener('click', getMLAccuracy);
    }
    if (document.getElementById('autoTuneBtn')) {
        document.getElementById('autoTuneBtn').addEventListener('click', autoTuneParameters);
    }
    if (document.getElementById('backtestBtn')) {
        document.getElementById('backtestBtn').addEventListener('click', runBacktest);
    }
    if (document.getElementById('forwardTestBtn')) {
        document.getElementById('forwardTestBtn').addEventListener('click', runForwardTest);
    }
    
    // Enter key in dropdowns
    document.getElementById('tradingPair').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') performAnalysis();
    });
    
    document.getElementById('timeframe').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') performAnalysis();
    });
}

/**
 * Update trading pairs dropdown based on selected category
 */
function updateTradingPairs(category) {
    const pairSelect = document.getElementById('tradingPair');
    pairSelect.innerHTML = '';
    
    if (tradingPairs[category]) {
        Object.keys(tradingPairs[category]).forEach(pair => {
            const option = document.createElement('option');
            option.value = tradingPairs[category][pair];
            option.textContent = pair;
            pairSelect.appendChild(option);
        });
    }
    
    console.log(`📊 Updated trading pairs for ${category}`);
}

/**
 * Perform Elliott Wave analysis
 */
async function performAnalysis() {
    if (isAnalyzing) {
        console.log('⏳ Analysis already in progress...');
        return;
    }
    
    const symbol = document.getElementById('tradingPair').value;
    const timeframe = document.getElementById('timeframe').value;
    
    if (!symbol) {
        showNotification('Please select a trading pair', 'warning');
        return;
    }
    
    console.log(`🔍 Analyzing ${symbol} on ${timeframe} timeframe`);
    
    // UI updates
    setAnalyzing(true);
    hideWelcomeMessage();
    showLoadingState();
    
    try {
        // Add timeout and better error handling
        const controller = new AbortController();
        // Dynamically build API URL based on current port
        const apiUrl = `${window.location.protocol}//${window.location.hostname}:${window.location.port}/api/analyze`;
        console.log('🌐 API URL:', apiUrl);
        // Set timeout for fetch (e.g., 15 seconds)
        const timeoutMs = 15000;
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                timeframe: timeframe
            }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.success) {
            console.log(`✅ Analysis complete: ${data.waves.length} waves detected`);
            currentAnalysis = data;
            displayAnalysisResults(data);
        } else {
            console.error('❌ Analysis failed:', data.error);
            showNotification(data.error || 'Analysis failed', 'error');
            showWelcomeMessage();
        }
    } catch (error) {
        console.error('🚨 Network error:', error);
        let errorMessage = 'Network error. Please check your connection.';
        if (error.name === 'AbortError') {
            errorMessage = 'Request timeout. The analysis is taking too long. Please try again.';
        } else if (error.message.includes('HTTP')) {
            errorMessage = `Server error: ${error.message}`;
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Unable to connect to server. Please ensure the server is running.';
        }
        showNotification(errorMessage, 'error');
        showWelcomeMessage();
    } finally {
        setAnalyzing(false);
        hideLoadingState();
    }
}

/**
 * Display analysis results with smart layout management
 */
function displayAnalysisResults(data) {
    // Update chart title with data-only mode indication
    const symbol = document.getElementById('tradingPair').selectedOptions[0].textContent;
    const timeframe = document.getElementById('timeframe').selectedOptions[0].textContent;
    document.getElementById('chartTitle').innerHTML = 
        `<i class="fas fa-table me-2"></i>${symbol}, ${timeframe} • Elliott Wave Data Analysis`;
    
    // Smart Chart Display Management
    handleChartDisplay(data);
    
    // Display market summary
    displayMarketSummary(data.market_summary);
    
    // Display Support/Resistance levels
    displaySupportResistanceLevels(data.support_resistance);
    
    // Smart layout management for analysis sections
    handleAnalysisResultsLayout(data);
    
    // Display future pattern predictions
    displayFuturePredictions(data.future_predictions);
    
    // Show analysis results section only if there's meaningful data
    const hasAnyData = data.waves.length > 0 || 
                      data.fibonacci_levels.length > 0 || 
                      (data.target_zones && data.target_zones.length > 0);
    
    if (hasAnyData) {
        document.getElementById('analysisResults').classList.remove('d-none');
        
        // Scroll to results with smooth animation
        setTimeout(() => {
            document.getElementById('analysisResults').scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }, 300);
    }
    
    // Show success notification with intelligent summary
    const successMessage = generateAnalysisSummary(data);
    showNotification(successMessage, 'success');
}

/**
 * Smart Chart Display Management
 */
function handleChartDisplay(data) {
    const chartContainer = document.getElementById('chartContainer');
    const noChartContainer = document.getElementById('noChartDataContainer');
    
    // Hide both containers initially
    chartContainer.classList.add('d-none');
    noChartContainer.classList.add('d-none');
    
    if (data.data_mode && data.ascii_table) {
        // DATA-ONLY MODE: Display ASCII table instead of chart
        displayASCIITable(data);
        
        // Display Technical Confluence Analysis immediately after ASCII table
        if (data.target_zones && data.target_zones.length > 0) {
            displayConfluenceAnalysis(data.target_zones, data.confluence_summary);
        }
        
    } else if (data.chart && data.chart.trim() !== '') {
        // CHART MODE: Display interactive chart
        try {
            const chartData = JSON.parse(data.chart);
            if (chartData && chartData.data && chartData.data.length > 0) {
                chartContainer.classList.remove('d-none');
                Plotly.newPlot('chartDiv', chartData.data, chartData.layout, {responsive: true});
                console.log('✅ Interactive chart displayed successfully');
            } else {
                throw new Error('Empty chart data');
            }
        } catch (error) {
            console.warn('⚠️ Chart parsing failed:', error);
            noChartContainer.classList.remove('d-none');
        }
        
    } else {
        // NO CHART DATA: Show informative message
        noChartContainer.classList.remove('d-none');
        console.log('ℹ️ No chart data available, showing message container');
    }
}

/**
 * Display ASCII Table for data-only mode
 */
function displayASCIITable(data) {
    // Create or update ASCII table container
    let asciiContainer = document.getElementById('asciiTableContainer');
    if (!asciiContainer) {
        asciiContainer = document.createElement('div');
        asciiContainer.id = 'asciiTableContainer';
        asciiContainer.className = 'card bg-dark border-info mb-4';
        asciiContainer.innerHTML = `
            <div class="card-header bg-gradient text-white">
                <h5 class="card-title mb-0">
                    <i class="fas fa-table me-2"></i>Elliott Wave Analysis Results
                </h5>
            </div>
            <div class="card-body">
                <pre id="asciiTable" class="text-light" style="font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.4; overflow-x: auto;"></pre>
            </div>
        `;
        
        // Insert after chart container
        const chartContainer = document.getElementById('chartContainer');
        chartContainer.parentNode.insertBefore(asciiContainer, chartContainer.nextSibling);
    }
    
    // Update ASCII table content
    document.getElementById('asciiTable').textContent = data.ascii_table;
    asciiContainer.classList.remove('d-none');
}

/**
 * Smart Analysis Results Layout Management
 */
function handleAnalysisResultsLayout(data) {
    // Handle Waves Section
    handleWavesSection(data.waves);
    
    // Handle Fibonacci Section
    handleFibonacciSection(data.fibonacci_levels);
    
    // Handle Confluence Section
    handleConfluenceSection(data.target_zones, data.confluence_summary);
    
    // Handle Validation Results
    if (data.validation_results) {
        displayValidationResults(data.validation_results);
    }
    
    // Adaptive column layout based on content availability
    adaptiveColumnLayout(data);
}

/**
 * Handle Waves Section Display
 */
function handleWavesSection(waves) {
    const waveCountBadge = document.getElementById('waveCountBadge');
    const noWavesMessage = document.getElementById('noWavesMessage');
    const wavesTableContainer = document.getElementById('wavesTableContainer');
    
    waveCountBadge.textContent = waves.length;
    
    if (waves.length === 0) {
        noWavesMessage.classList.remove('d-none');
        wavesTableContainer.classList.add('d-none');
    } else {
        noWavesMessage.classList.add('d-none');
        wavesTableContainer.classList.remove('d-none');
        displayWaveData(waves);
    }
}

/**
 * Handle Fibonacci Section Display
 */
function handleFibonacciSection(fibonacciLevels) {
    const fibCountBadge = document.getElementById('fibCountBadge');
    const noFibonacciMessage = document.getElementById('noFibonacciMessage');
    const fibonacciContainer = document.getElementById('fibonacciLevels');
    
    fibCountBadge.textContent = fibonacciLevels.length;
    
    if (fibonacciLevels.length === 0) {
        noFibonacciMessage.classList.remove('d-none');
        fibonacciContainer.innerHTML = '';
    } else {
        noFibonacciMessage.classList.add('d-none');
        displayFibonacciLevels(fibonacciLevels);
    }
}

/**
 * Handle Confluence Section Display
 */
function handleConfluenceSection(targetZones, confluenceSummary) {
    const confluenceResults = document.getElementById('confluenceResults');
    const confluenceCountBadge = document.getElementById('confluenceCountBadge');
    const noConfluenceMessage = document.getElementById('noConfluenceMessage');
    const confluenceContent = document.getElementById('confluenceContent');
    
    const targetCount = targetZones ? targetZones.length : 0;
    confluenceCountBadge.textContent = `${targetCount} target${targetCount !== 1 ? 's' : ''}`;
    
    if (targetCount === 0) {
        noConfluenceMessage.classList.remove('d-none');
        confluenceContent.classList.add('d-none');
        confluenceResults.classList.remove('d-none'); // Still show the section with the message
    } else {
        noConfluenceMessage.classList.add('d-none');
        confluenceContent.classList.remove('d-none');
        confluenceResults.classList.remove('d-none');
        displayConfluenceAnalysis(targetZones, confluenceSummary);
    }
}

/**
 * Adaptive Column Layout Based on Content
 */
function adaptiveColumnLayout(data) {
    const wavesColumn = document.getElementById('wavesResultsColumn');
    const fibonacciColumn = document.getElementById('fibonacciResultsColumn');
    
    const hasWaves = data.waves.length > 0;
    const hasFibonacci = data.fibonacci_levels.length > 0;
    
    // Reset classes
    wavesColumn.className = '';
    fibonacciColumn.className = '';
    
    if (hasWaves && hasFibonacci) {
        // Both sections have data - use normal 50/50 layout
        wavesColumn.className = 'col-md-6';
        fibonacciColumn.className = 'col-md-6';
    } else if (hasWaves && !hasFibonacci) {
        // Only waves - expand waves column, hide fibonacci
        wavesColumn.className = 'col-12';
        fibonacciColumn.classList.add('d-none');
    } else if (!hasWaves && hasFibonacci) {
        // Only fibonacci - hide waves, expand fibonacci
        wavesColumn.classList.add('d-none');
        fibonacciColumn.className = 'col-12';
    } else {
        // Neither section has data - use normal layout but show empty messages
        wavesColumn.className = 'col-md-6';
        fibonacciColumn.className = 'col-md-6';
    }
}

/**
 * Generate intelligent analysis summary
 */
function generateAnalysisSummary(data) {
    const waveCount = data.waves.length;
    const fibCount = data.fibonacci_levels.length;
    const targetCount = data.target_zones ? data.target_zones.length : 0;
    const predictionCount = data.future_predictions.length;
    
    let parts = [];
    
    if (waveCount > 0) {
        parts.push(`${waveCount} wave${waveCount !== 1 ? 's' : ''}`);
    }
    
    if (fibCount > 0) {
        parts.push(`${fibCount} Fibonacci level${fibCount !== 1 ? 's' : ''}`);
    }
    
    if (targetCount > 0) {
        parts.push(`${targetCount} confluence target${targetCount !== 1 ? 's' : ''}`);
    }
    
    if (predictionCount > 0) {
        parts.push(`${predictionCount} prediction${predictionCount !== 1 ? 's' : ''}`);
    }
    
    if (parts.length === 0) {
        return 'Analysis complete. Limited data available for current selection.';
    }
    
    const analysisTime = data.analysis_time ? ` in ${data.analysis_time}s` : '';
    return `Analysis complete${analysisTime}! Found: ${parts.join(', ')}.`;
}

/**
 * Scroll to analysis results (helper function)
 */
function scrollToAnalysisResults() {
    const analysisResults = document.getElementById('analysisResults');
    if (!analysisResults.classList.contains('d-none')) {
        analysisResults.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
}

/**
 * Display market summary
 */
function displayMarketSummary(summary) {
    const summaryHtml = `
        <div class="market-stat">
            <span class="label">Symbol:</span>
            <span class="value">${summary.symbol}</span>
        </div>
        <div class="market-stat">
            <span class="label">Current Price:</span>
            <span class="value">$${summary.current_price.toLocaleString()}</span>
        </div>
        <div class="market-stat">
            <span class="label">24h Change:</span>
            <span class="value ${summary.change_24h >= 0 ? 'positive' : 'negative'}">
                ${summary.change_24h >= 0 ? '+' : ''}${summary.change_24h}%
            </span>
        </div>
        <div class="market-stat">
            <span class="label">52W High:</span>
            <span class="value">$${summary.high_52w.toLocaleString()}</span>
        </div>
        <div class="market-stat">
            <span class="label">52W Low:</span>
            <span class="value">$${summary.low_52w.toLocaleString()}</span>
        </div>
        <div class="market-stat">
            <span class="label">Data Points:</span>
            <span class="value">${summary.data_points}</span>
        </div>
        <div class="market-stat">
            <span class="label">Last Update:</span>
            <span class="value">${formatDateTime(summary.last_update)}</span>
        </div>
    `;
    
    document.getElementById('marketSummary').innerHTML = summaryHtml;
    document.getElementById('marketSummaryCard').classList.remove('d-none');
}

/**
 * Display Support/Resistance levels with priority system and distance info
 */
function displaySupportResistanceLevels(srData) {
    if (!srData) return;
    
    let levelsHtml = '';
    
    // Support levels with priority
    if (srData.support_levels && srData.support_levels.length > 0) {
        srData.support_levels.forEach((level, index) => {
            const priority = level.priority || 'normal';
            const distance = level.distance_percent ? level.distance_percent.toFixed(1) : '0.0';
            let priorityIcon = '🔰';
            let priorityText = '';
            let priorityClass = 'sr-support';
            
            if (priority === 'nearest') {
                priorityIcon = '🎯';
                priorityText = ' (NEAREST)';
                priorityClass = 'sr-support-nearest';
            } else if (priority === 'strongest') {
                priorityIcon = '💪';
                priorityText = ' (STRONGEST)';
                priorityClass = 'sr-support-strongest';
            }
            
            levelsHtml += `
                <div class="sr-level-item ${priorityClass}">
                    <span class="sr-support">
                        ${priorityIcon} Support ${index + 1}${priorityText}
                        <small class="text-muted d-block">${distance}% away</small>
                    </span>
                    <span class="sr-support">$${level.price.toFixed(4)}</span>
                </div>
            `;
        });
    }
    
    // Resistance levels with priority
    if (srData.resistance_levels && srData.resistance_levels.length > 0) {
        srData.resistance_levels.forEach((level, index) => {
            const priority = level.priority || 'normal';
            const distance = level.distance_percent ? level.distance_percent.toFixed(1) : '0.0';
            let priorityIcon = '🔴';
            let priorityText = '';
            let priorityClass = 'sr-resistance';
            
            if (priority === 'nearest') {
                priorityIcon = '🎯';
                priorityText = ' (NEAREST)';
                priorityClass = 'sr-resistance-nearest';
            } else if (priority === 'strongest') {
                priorityIcon = '💪';
                priorityText = ' (STRONGEST)';
                priorityClass = 'sr-resistance-strongest';
            }
            
            levelsHtml += `
                <div class="sr-level-item ${priorityClass}">
                    <span class="sr-resistance">
                        ${priorityIcon} Resistance ${index + 1}${priorityText}
                        <small class="text-muted d-block">${distance}% away</small>
                    </span>
                    <span class="sr-resistance">$${level.price.toFixed(4)}</span>
                </div>
            `;
        });
    }
    
    // Psychological levels (limited to first 3)
    if (srData.psychological_levels && srData.psychological_levels.length > 0) {
        const visiblePsychLevels = srData.psychological_levels.slice(0, 3);
        visiblePsychLevels.forEach((level, index) => {
            levelsHtml += `
                <div class="sr-level-item">
                    <span class="sr-psychological">
                        💰 Psychological ${index + 1}
                        <small class="text-muted d-block">Round number</small>
                    </span>
                    <span class="sr-psychological">$${level.price.toFixed(0)}</span>
                </div>
            `;
        });
    }
    
    // Add summary info if available
    if (srData.analysis_summary) {
        const summary = srData.analysis_summary;
        levelsHtml += `
            <div class="sr-summary mt-3 p-2 border-top">
                <small class="text-muted">
                    📊 Analysis: ${summary.total_support_found} supports, ${summary.total_resistance_found} resistances found
                </small>
            </div>
        `;
    }
    
    if (levelsHtml) {
        document.getElementById('srLevelsList').innerHTML = levelsHtml;
        document.getElementById('srLevelsCard').classList.remove('d-none');
    } else {
        document.getElementById('srLevelsCard').classList.add('d-none');
    }
}

/**
 * Display wave data in table
 */
function displayWaveData(waves) {
    const tbody = document.querySelector('#wavesTable tbody');
    tbody.innerHTML = '';
    
    if (waves.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">
                    <i class="fas fa-info-circle me-2"></i>
                    No Elliott Waves detected for this timeframe. Try a different timeframe or symbol.
                </td>
            </tr>
        `;
        return;
    }
    
    waves.forEach(wave => {
        const row = document.createElement('tr');
        
        const directionClass = wave.direction === 'UP' ? 'wave-up' : 'wave-down';
        const directionIcon = wave.direction === 'UP' ? 'fa-arrow-up' : 'fa-arrow-down';
        
        const confidenceClass = getConfidenceClass(wave.confidence);
        
        row.innerHTML = `
            <td>
                <strong>${wave.type.replace('WAVE_', '')}</strong>
            </td>
            <td class="${directionClass}">
                <i class="fas ${directionIcon} me-1"></i>
                ${wave.direction}
            </td>
            <td>$${wave.start_price.toLocaleString()}</td>
            <td>$${wave.end_price.toLocaleString()}</td>
            <td class="${wave.price_change >= 0 ? 'positive' : 'negative'}">
                ${wave.price_change >= 0 ? '+' : ''}${wave.price_change}%
            </td>
            <td class="${confidenceClass}">
                ${(wave.confidence * 100).toFixed(1)}%
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Display Fibonacci levels with enhanced visualization
 */
function displayFibonacciLevels(fibonacciData) {
    const container = document.getElementById('fibonacciLevels');
    container.innerHTML = '';
    
    if (!fibonacciData || fibonacciData.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-info-circle me-2"></i>
                No Fibonacci levels calculated. Need corrective waves (2, 4, B) for analysis.
            </div>
        `;
        return;
    }
    
    fibonacciData.forEach(fibData => {
        const waveDiv = document.createElement('div');
        waveDiv.className = 'fibonacci-wave mb-4 p-3 rounded';
        waveDiv.style.background = 'rgba(255,255,255,0.05)';
        waveDiv.style.border = '1px solid rgba(255,255,255,0.1)';

        // Safely get wave number
        let waveNumber = 'Unknown';
        if (fibData.wave && typeof fibData.wave === 'string') {
            waveNumber = fibData.wave.replace('WAVE_', '');
        }
        const directionIcon = fibData.direction === 'UP' ? 'fa-arrow-up text-success' : 'fa-arrow-down text-danger';

        let levelsHtml = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h6 class="mb-0">
                    <i class="fas fa-wave-square me-2"></i>Wave ${waveNumber} Retracements
                    <i class="fas ${directionIcon} ms-2"></i>
                </h6>
                <span class="badge bg-info">${(fibData.confidence * 100).toFixed(1)}% confidence</span>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <small class="text-muted">Price Range: $${fibData.start_price?.toFixed(4) ?? 'N/A'} → $${fibData.end_price?.toFixed(4) ?? 'N/A'}</small>
                </div>
            </div>
            <div class="fibonacci-levels-grid mt-3">
        `;

        // Create a responsive grid for Fibonacci levels
        if (fibData.levels) {
            Object.entries(fibData.levels).forEach(([level, price]) => {
                const percentage = parseFloat(level.replace('%', ''));
                let levelClass = 'fibonacci-level-normal';

                // Highlight key Fibonacci levels
                if ([38.2, 50.0, 61.8].includes(percentage)) {
                    levelClass = 'fibonacci-level-key';
                } else if ([23.6, 78.6].includes(percentage)) {
                    levelClass = 'fibonacci-level-secondary';
                } else if ([100, 161.8].includes(percentage)) {
                    levelClass = 'fibonacci-level-extension';
                }

                levelsHtml += `
                    <div class="fibonacci-level ${levelClass} d-flex justify-content-between align-items-center p-2 mb-1 rounded">
                        <span class="level-label fw-bold">${level}</span>
                        <span class="level-price">$${price?.toFixed(4) ?? 'N/A'}</span>
                    </div>
                `;
            });
        }

        levelsHtml += '</div>';
        waveDiv.innerHTML = levelsHtml;
        container.appendChild(waveDiv);
    });
}

/**
 * Display validation results
 */
function displayValidationResults(validationResults) {
    // Show validation results section
    document.getElementById('validationResults').classList.remove('d-none');
    
    // Update overall validation score
    const overallScore = Math.round(validationResults.overall_score * 100);
    const progressBar = document.getElementById('overallValidationScore');
    const scoreText = document.getElementById('validationScoreText');
    
    progressBar.style.width = `${overallScore}%`;
    scoreText.textContent = `${overallScore}%`;
    
    // Set progress bar color based on score
    progressBar.className = 'progress-bar';
    if (overallScore >= 80) {
        progressBar.classList.add('bg-success');
    } else if (overallScore >= 60) {
        progressBar.classList.add('bg-warning');
    } else {
        progressBar.classList.add('bg-danger');
    }
    
    // Update validation summary
    const summaryContainer = document.getElementById('validationSummary');
    let summaryHtml = '';
    
    if (validationResults.summary) {
        summaryHtml += `<div class="alert alert-info">
            <strong>Analysis Summary:</strong><br>
            ${validationResults.summary}
        </div>`;
    }
    
    if (validationResults.recommendations && validationResults.recommendations.length > 0) {
        summaryHtml += '<h6>Recommendations:</h6><ul class="list-unstyled">';
        validationResults.recommendations.forEach(rec => {
            summaryHtml += `<li><i class="fas fa-lightbulb text-warning me-2"></i>${rec}</li>`;
        });
        summaryHtml += '</ul>';
    }
    
    summaryContainer.innerHTML = summaryHtml;
    
    // Update validation table
    const tableBody = document.querySelector('#validationTable tbody');
    tableBody.innerHTML = '';
    
    if (validationResults.rule_details) {
        Object.entries(validationResults.rule_details).forEach(([rule, details]) => {
            const row = document.createElement('tr');
            const score = Math.round(details.score * 100);
            
            let statusIcon = '';
            let statusClass = '';
            if (score >= 80) {
                statusIcon = '<i class="fas fa-check-circle text-success"></i>';
                statusClass = 'text-success';
            } else if (score >= 60) {
                statusIcon = '<i class="fas fa-exclamation-triangle text-warning"></i>';
                statusClass = 'text-warning';
            } else {
                statusIcon = '<i class="fas fa-times-circle text-danger"></i>';
                statusClass = 'text-danger';
            }
            
            row.innerHTML = `
                <td>${rule.replace(/_/g, ' ').toUpperCase()}</td>
                <td>${statusIcon}</td>
                <td class="${statusClass}">${score}%</td>
            `;
            
            // Add tooltip with details if available
            if (details.details) {
                row.title = details.details;
            }
            
            tableBody.appendChild(row);
        });
    }
}

/**
 * Display future pattern predictions
 */
function displayFuturePredictions(predictions) {
    const container = document.getElementById('futurePredictions') || createFuturePredictionsContainer();
    container.innerHTML = '';
    
    if (!predictions || predictions.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-crystal-ball me-2"></i>
                No future predictions available. Need more wave data for analysis.
            </div>
        `;
        return;
    }
    
    predictions.forEach(prediction => {
        const predictionDiv = document.createElement('div');
        predictionDiv.className = 'prediction-card mb-4 p-4 rounded';
        predictionDiv.style.background = 'linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,165,0,0.05))';
        predictionDiv.style.border = '2px solid rgba(255,215,0,0.3)';
        predictionDiv.style.boxShadow = '0 4px 15px rgba(255,215,0,0.1)';
        
        let probabilityColor = 'warning';
        if (prediction.probability.includes('High')) probabilityColor = 'success';
        else if (prediction.probability.includes('Low')) probabilityColor = 'danger';
        
        let predictionsHtml = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="mb-0 text-warning">
                    <i class="fas fa-crystal-ball me-2"></i>${prediction.pattern}
                </h5>
                <span class="badge bg-${probabilityColor} fs-6">${prediction.probability}</span>
            </div>
            <p class="text-light mb-3">${prediction.expected_move}</p>
            <div class="row">
        `;
        
        prediction.targets.forEach((target, index) => {
            const targetClass = index === 0 ? 'border-success' : 'border-info';
            predictionsHtml += `
                <div class="col-md-6 mb-3">
                    <div class="target-card p-3 rounded border ${targetClass}">
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="fw-bold">${target.level}</span>
                            <span class="text-success fs-5">$${target.price.toFixed(4)}</span>
                        </div>
                        <small class="text-muted">${target.ratio}</small>
                    </div>
                </div>
            `;
        });
        
        predictionsHtml += '</div>';
        predictionDiv.innerHTML = predictionsHtml;
        container.appendChild(predictionDiv);
    });
}

/**
 * Create future predictions container if it doesn't exist
 */
function createFuturePredictionsContainer() {
    // Find a good place to insert the predictions container
    const fibContainer = document.getElementById('fibonacciCard');
    if (fibContainer) {
        const predictionsCard = document.createElement('div');
        predictionsCard.id = 'futurePredictionsCard';
        predictionsCard.className = 'card bg-dark border-warning mb-4';
        predictionsCard.innerHTML = `
            <div class="card-header bg-gradient text-white">
                <h5 class="card-title mb-0">
                    <i class="fas fa-crystal-ball me-2"></i>Future Pattern Predictions
                </h5>
            </div>
            <div class="card-body">
                <div id="futurePredictions"></div>
            </div>
        `;
        
        // Insert after Fibonacci card
        fibContainer.parentNode.insertBefore(predictionsCard, fibContainer.nextSibling);
        return document.getElementById('futurePredictions');
    }
    
    // Fallback: create a simple container
    const container = document.createElement('div');
    container.id = 'futurePredictions';
    document.getElementById('analysisResults').appendChild(container);
    return container;
}

/**
 * Get confidence class for styling
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

/**
 * Format date and time
 */
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

/**
 * Set analyzing state
 */
function setAnalyzing(analyzing) {
    isAnalyzing = analyzing;
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (analyzing) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search-plus me-2"></i>Analyze Waves';
    }
}

/**
 * Show/hide loading state
 */
function showLoadingState() {
    document.getElementById('loadingSpinner').classList.remove('d-none');
}

function hideLoadingState() {
    document.getElementById('loadingSpinner').classList.add('d-none');
}

/**
 * Show/hide welcome message
 */
function showWelcomeMessage() {
    document.getElementById('welcomeMessage').classList.remove('d-none');
    document.getElementById('chartContainer').classList.add('d-none');
    document.getElementById('analysisResults').classList.add('d-none');
    document.getElementById('validationResults').classList.add('d-none');
    document.getElementById('marketSummaryCard').classList.add('d-none');
}

function hideWelcomeMessage() {
    document.getElementById('welcomeMessage').classList.add('d-none');
}

/**
 * Export chart functionality
 */
function exportChart() {
    if (!currentAnalysis) {
        showNotification('No chart to export', 'warning');
        return;
    }
    
    try {
        const symbol = document.getElementById('tradingPair').selectedOptions[0].textContent;
        const timeframe = document.getElementById('timeframe').value;
        const filename = `${symbol}_${timeframe}_elliott_waves_${new Date().getTime()}.png`;
        
        Plotly.downloadImage('chartDiv', {
            format: 'png',
            width: 1200,
            height: 700,
            filename: filename
        });
        
        showNotification('Chart exported successfully!', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Failed to export chart', 'error');
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${getBootstrapAlertClass(type)} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    
    const icon = getNotificationIcon(type);
    
    notification.innerHTML = `
        <i class="${icon} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Get ML-based wave accuracy prediction
 */
function getMLAccuracy() {
    if (!mlFeaturesAvailable) {
        showNotification('ML features not available', 'warning');
        return;
    }
    
    const symbol = document.getElementById('tradingPair').value;
    
    showNotification('🤖 Analyzing wave accuracy with ML...', 'info');
    
    fetch('/api/ml/accuracy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification(`❌ ML Accuracy Error: ${data.error}`, 'error');
            return;
        }
        
        // Display accuracy results
        const accuracyHtml = `
            <div class="ml-accuracy-results">
                <h4>🎯 ML Wave Accuracy Analysis</h4>
                <p><strong>Symbol:</strong> ${data.symbol}</p>
                <p><strong>Accuracy Score:</strong> ${(data.accuracy_score * 100).toFixed(1)}%</p>
                <p><strong>Confidence Level:</strong> ${data.confidence_level}</p>
                <p><strong>Pattern Match Score:</strong> ${(data.pattern_match_score * 100).toFixed(1)}%</p>
                ${data.similar_patterns.length > 0 ? 
                    `<p><strong>Similar Patterns Found:</strong> ${data.similar_patterns.length}</p>` : ''}
            </div>
        `;
        
        // Add to results area
        const resultsDiv = document.getElementById('results');
        const accuracyDiv = document.createElement('div');
        accuracyDiv.innerHTML = accuracyHtml;
        resultsDiv.appendChild(accuracyDiv);
        
        showNotification(`✅ Wave accuracy: ${(data.accuracy_score * 100).toFixed(1)}%`, 'success');
    })
    .catch(error => {
        console.error('Error getting ML accuracy:', error);
        showNotification('❌ Error analyzing wave accuracy', 'error');
    });
}

/**
 * Auto-tune Elliott Wave parameters
 */
function autoTuneParameters() {
    if (!mlFeaturesAvailable) {
        showNotification('Auto-tuning not available', 'warning');
        return;
    }
    
    const symbol = document.getElementById('tradingPair').value;
    const timeframe = document.getElementById('timeframe').value;
    
    showNotification('⚙️ Auto-tuning parameters...', 'info');
    
    fetch('/api/ml/auto-tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol, timeframe: timeframe })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification(`❌ Auto-tuning Error: ${data.error}`, 'error');
            return;
        }
        
        // Display tuning results
        const tuningHtml = `
            <div class="auto-tuning-results">
                <h4>⚙️ Auto-Tuning Results</h4>
                <p><strong>Symbol:</strong> ${data.symbol} (${data.timeframe})</p>
                <p><strong>Optimal Threshold:</strong> ${data.optimal_threshold.toFixed(4)}</p>
                <p><strong>Optimal Wave Length:</strong> ${data.optimal_min_wave_length}</p>
                <p><strong>Optimal Lookback:</strong> ${data.optimal_lookback_periods}</p>
                <p><strong>Confidence Score:</strong> ${(data.confidence_score * 100).toFixed(1)}%</p>
                <p><strong>Multi-TF Confirmed:</strong> ${data.multi_timeframe_confirmed ? '✅' : '❌'}</p>
                <p><em>${data.status}</em></p>
            </div>
        `;
        
        // Add to results area
        const resultsDiv = document.getElementById('results');
        const tuningDiv = document.createElement('div');
        tuningDiv.innerHTML = tuningHtml;
        resultsDiv.appendChild(tuningDiv);
        
        showNotification('✅ Parameters optimized and applied', 'success');
    })
    .catch(error => {
        console.error('Error auto-tuning:', error);
        showNotification('❌ Error auto-tuning parameters', 'error');
    });
}

/**
 * Run comprehensive backtesting
 */
function runBacktest() {
    if (!mlFeaturesAvailable) {
        showNotification('Backtesting not available', 'warning');
        return;
    }
    
    const symbol = document.getElementById('tradingPair').value;
    
    showNotification('📊 Running backtest...', 'info');
    
    fetch('/api/ml/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification(`❌ Backtest Error: ${data.error}`, 'error');
            return;
        }
        
        // Display backtest results
        const backtestHtml = `
            <div class="backtest-results">
                <h4>📊 Backtest Results</h4>
                <p><strong>Symbol:</strong> ${data.symbol}</p>
                <p><strong>Total Trades:</strong> ${data.total_trades}</p>
                <p><strong>Win Rate:</strong> ${data.win_rate.toFixed(1)}%</p>
                <p><strong>Total Return:</strong> ${data.total_return.toFixed(2)}%</p>
                <p><strong>Max Drawdown:</strong> ${data.max_drawdown.toFixed(2)}%</p>
                <p><strong>Sharpe Ratio:</strong> ${data.sharpe_ratio.toFixed(2)}</p>
                <p><strong>Profit Factor:</strong> ${data.profit_factor.toFixed(2)}</p>
                <p><strong>Confidence Score:</strong> ${(data.confidence_score * 100).toFixed(1)}%</p>
                <details>
                    <summary>📋 Detailed Report</summary>
                    <pre>${data.detailed_report}</pre>
                </details>
            </div>
        `;
        
        // Add to results area
        const resultsDiv = document.getElementById('results');
        const backtestDiv = document.createElement('div');
        backtestDiv.innerHTML = backtestHtml;
        resultsDiv.appendChild(backtestDiv);
        
        showNotification(`✅ Backtest complete: ${data.win_rate.toFixed(1)}% win rate`, 'success');
    })
    .catch(error => {
        console.error('Error running backtest:', error);
        showNotification('❌ Error running backtest', 'error');
    });
}

/**
 * Run forward-walking validation
 */
function runForwardTest() {
    if (!mlFeaturesAvailable) {
        showNotification('Forward testing not available', 'warning');
        return;
    }
    
    const symbol = document.getElementById('tradingPair').value;
    
    showNotification('🔄 Running forward-walking validation...', 'info');
    
    fetch('/api/ml/forward-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: symbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification(`❌ Forward Test Error: ${data.error}`, 'error');
            return;
        }
        
        // Display forward test results
        const forwardTestHtml = `
            <div class="forward-test-results">
                <h4>🔄 Forward-Walking Validation</h4>
                <p><strong>Symbol:</strong> ${data.symbol}</p>
                <p><strong>Test Periods:</strong> ${data.num_periods}</p>
                <p><strong>Avg Win Rate:</strong> ${data.avg_win_rate.toFixed(1)}%</p>
                <p><strong>Avg Profit Factor:</strong> ${data.avg_profit_factor.toFixed(2)}</p>
                <p><strong>Avg Sharpe Ratio:</strong> ${data.avg_sharpe_ratio.toFixed(2)}</p>
                <p><strong>Avg Max Drawdown:</strong> ${data.avg_max_drawdown.toFixed(2)}%</p>
                <p><strong>Consistency Score:</strong> ${data.consistency_score.toFixed(2)} (lower is better)</p>
                <p><strong>Avg Confidence:</strong> ${(data.avg_confidence * 100).toFixed(1)}%</p>
                <p><em>${data.status}</em></p>
            </div>
        `;
        
        // Add to results area
        const resultsDiv = document.getElementById('results');
        const forwardTestDiv = document.createElement('div');
        forwardTestDiv.innerHTML = forwardTestHtml;
        resultsDiv.appendChild(forwardTestDiv);
        
        showNotification('✅ Forward validation complete', 'success');
    })
    .catch(error => {
        console.error('Error running forward test:', error);
        showNotification('❌ Error running forward validation', 'error');
    });
}

/**
 * Get Bootstrap alert class for notification type
 */
function getBootstrapAlertClass(type) {
    const mapping = {
        'success': 'success',
        'error': 'danger',
        'warning': 'warning',
        'info': 'info'
    };
    return mapping[type] || 'info';
}

/**
 * Get notification icon
 */
function getNotificationIcon(type) {
    const mapping = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'warning': 'fas fa-exclamation-triangle',
        'info': 'fas fa-info-circle'
    };
    return mapping[type] || 'fas fa-info-circle';
}

/**
 * Display Technical Confluence Analysis Results
 */
function displayConfluenceAnalysis(targetZones, confluenceSummary) {
    console.log('📊 Displaying Technical Confluence Analysis...');
    
    // Show confluence results section prominently
    const confluenceSection = document.getElementById('confluenceResults');
    confluenceSection.classList.remove('d-none');
    
    // Add enhanced styling for better visual prominence
    confluenceSection.style.marginTop = '2rem';
    confluenceSection.style.marginBottom = '2rem';
    
    // Display confluence summary with enhanced formatting
    if (confluenceSummary) {
        const summaryDiv = document.getElementById('confluenceSummary');
        const bestTarget = confluenceSummary.best_target;
        
        summaryDiv.innerHTML = `
            <div class="row text-center">
                <div class="col-md-2">
                    <div class="stat-card">
                        <h4 class="text-primary">${confluenceSummary.total_targets}</h4>
                        <small class="text-muted">Total Targets</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="stat-card">
                        <h4 class="text-success">${confluenceSummary.high_confidence}</h4>
                        <small class="text-muted">High Confidence</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="stat-card">
                        <h4 class="text-warning">${confluenceSummary.medium_confidence}</h4>
                        <small class="text-muted">Medium Confidence</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="stat-card">
                        <h4 class="text-secondary">${confluenceSummary.low_confidence}</h4>
                        <small class="text-muted">Low Confidence</small>
                    </div>
                </div>
                <div class="col-md-4">
                    ${bestTarget ? `
                    <div class="best-target-card p-3 bg-light rounded">
                        <strong>🎯 Best Target:</strong><br>
                        <span class="badge bg-primary fs-6">${bestTarget.wave}</span><br>
                        <strong class="text-success">$${bestTarget.price.toFixed(4)}</strong><br>
                        <small class="text-muted">${bestTarget.confidence} • ${(bestTarget.probability * 100).toFixed(0)}% • ${bestTarget.confluences} confluences</small>
                    </div>
                    ` : '<div class="text-muted">No best target available</div>'}
                </div>
            </div>
        `;
    }
    
    // Display target zones table with enhanced formatting
    const tableBody = document.querySelector('#targetZonesTable tbody');
    tableBody.innerHTML = '';
    
    targetZones.forEach((zone, index) => {
        const row = document.createElement('tr');
        
        // Store zone data on the row for easy access
        row.dataset.zoneIndex = index;
        
        // Determine row class based on confidence level with enhanced styling
        let rowClass = '';
        if (zone.confidence_level === 'HIGH') rowClass = 'table-success';
        else if (zone.confidence_level === 'MEDIUM') rowClass = 'table-warning';
        else rowClass = 'table-light';
        
        row.className = rowClass;
        
        // Create confluences display (limit to 3 main ones for table)
        const confluencesDisplay = (zone.confluences || zone.all_confluences || []).slice(0, 3).map(conf => 
            `<span class="badge bg-info me-1" title="${conf}">${conf}</span>`
        ).join('');
        
        const totalConfluences = zone.all_confluences ? zone.all_confluences.length : (zone.confluences || []).length;
        const moreConfluences = totalConfluences > 3 ? 
            `<br><small class="text-muted">+${totalConfluences - 3} more</small>` : '';
        
        row.innerHTML = `
            <td>
                <div class="d-flex align-items-center">
                    <span class="badge bg-dark me-2">${index + 1}</span>
                    <div>
                        <strong>${zone.wave_target}</strong>
                        <br><small class="text-muted">Elliott Wave Target</small>
                    </div>
                </div>
            </td>
            <td>
                <strong class="fs-6">$${zone.price_level.toFixed(4)}</strong>
            </td>
            <td>
                <span class="badge fs-6 ${zone.price_change_pct >= 0 ? 'bg-success' : 'bg-danger'}">
                    ${zone.price_change_pct >= 0 ? '+' : ''}${zone.price_change_pct.toFixed(2)}%
                </span>
            </td>
            <td>
                <small class="text-wrap">${zone.elliott_basis}</small>
            </td>
            <td>
                <span class="badge fs-6 ${zone.confidence_level === 'HIGH' ? 'bg-success' : 
                                       zone.confidence_level === 'MEDIUM' ? 'bg-warning' : 'bg-secondary'}">
                    ${zone.confidence_level}
                </span>
            </td>
            <td>
                <strong class="text-primary">${(zone.probability * 100).toFixed(0)}%</strong>
            </td>
            <td>
                <span class="badge fs-6 ${zone.risk_reward_ratio >= 2 ? 'bg-success' : 
                                       zone.risk_reward_ratio >= 1 ? 'bg-warning' : 'bg-danger'}">
                    ${zone.risk_reward_ratio.toFixed(2)}:1
                </span>
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2" style="height: 25px; width: 80px;">
                        <div class="progress-bar ${zone.confluence_score >= 80 ? 'bg-success' : 
                                                  zone.confluence_score >= 60 ? 'bg-warning' : 'bg-danger'}" 
                             style="width: ${zone.confluence_score}%">
                             <small class="text-white fw-bold">${zone.confluence_score.toFixed(0)}</small>
                        </div>
                    </div>
                </div>
            </td>
            <td>
                <div class="confluence-list">
                    ${confluencesDisplay}
                    ${moreConfluences}
                    <br><button class="btn btn-sm btn-outline-info mt-1" 
                               onclick="showConfluenceModal(${index})" 
                               title="Click to see detailed explanations">
                        <i class="fas fa-info-circle me-1"></i>Why This Target?
                    </button>
                </div>
            </td>
        `;
        
        // Add enhanced click handler with visual feedback
        row.addEventListener('click', () => {
            // Remove previous selections
            document.querySelectorAll('#targetZonesTable tbody tr').forEach(r => r.classList.remove('table-primary'));
            // Highlight selected row
            row.classList.add('table-primary');
            showConfluenceDetails(zone, index);
        });
        
        row.style.cursor = 'pointer';
        row.title = 'Click for detailed confluence breakdown';
        
        // Add hover effect
        row.addEventListener('mouseenter', () => {
            if (!row.classList.contains('table-primary')) {
                row.style.backgroundColor = '#f8f9fa';
            }
        });
        
        row.addEventListener('mouseleave', () => {
            if (!row.classList.contains('table-primary')) {
                row.style.backgroundColor = '';
            }
        });
        
        tableBody.appendChild(row);
    });
    
    // Create detailed confluence breakdown with enhanced styling
    displayDetailedConfluences(targetZones);
    
    // Add custom CSS for better visual presentation
    addConfluenceCustomStyles();
}

/**
 * Add custom CSS styles for enhanced confluence display
 */
function addConfluenceCustomStyles() {
    // Check if styles already added
    if (document.getElementById('confluenceCustomStyles')) return;
    
    const style = document.createElement('style');
    style.id = 'confluenceCustomStyles';
    style.textContent = `
        .stat-card {
            padding: 1rem;
            border-radius: 8px;
            background: rgba(0,123,255,0.1);
            margin-bottom: 0.5rem;
        }
        .best-target-card {
            border: 2px solid #28a745;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .confluence-detail-card {
            transition: all 0.3s ease;
            border-left: 4px solid #dee2e6;
        }
        .confluence-detail-card:hover {
            border-left-color: #007bff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .confluence-detail-card.border-primary {
            border-left-color: #007bff;
            background-color: rgba(0,123,255,0.05);
        }
        .confluence-tags .badge {
            font-size: 0.75em;
            margin: 2px;
        }
        #targetZonesTable tbody tr {
            transition: background-color 0.2s ease;
        }
        #targetZonesTable tbody tr.table-primary {
            background-color: rgba(0,123,255,0.1) !important;
            border-left: 4px solid #007bff;
        }
    `;
    document.head.appendChild(style);
}

/**
 * Display detailed confluence breakdown for all target zones
 */
function displayDetailedConfluences(targetZones) {
    const detailsDiv = document.getElementById('confluenceDetails');
    
    const detailsHtml = targetZones.slice(0, 5).map((zone, index) => `
        <div class="card mb-3 confluence-detail-card" data-zone-index="${index}">
            <div class="card-header d-flex justify-content-between align-items-center bg-gradient">
                <h6 class="mb-0">
                    <i class="fas fa-target me-2 text-primary"></i>
                    <span class="badge bg-dark me-2">${index + 1}</span>
                    Target: ${zone.wave_target} 
                    <span class="badge fs-6 ${zone.confidence_level === 'HIGH' ? 'bg-success' : 
                                             zone.confidence_level === 'MEDIUM' ? 'bg-warning' : 'bg-secondary'}">
                        ${zone.confidence_level}
                    </span>
                </h6>
                <div class="text-end">
                    <div class="h5 mb-1 text-success">$${zone.price_level.toFixed(4)}</div>
                    <small class="text-muted">${zone.price_change_pct >= 0 ? '+' : ''}${zone.price_change_pct.toFixed(2)}%</small>
                </div>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <h6 class="text-primary"><i class="fas fa-chart-line me-2"></i>Elliott Wave Analysis</h6>
                        <ul class="list-unstyled ms-3">
                            <li><i class="fas fa-wave-square me-2 text-info"></i><strong>Basis:</strong> ${zone.elliott_basis}</li>
                            <li><i class="fas fa-percent me-2 text-success"></i><strong>Probability:</strong> ${(zone.probability * 100).toFixed(0)}%</li>
                            <li><i class="fas fa-balance-scale me-2 text-warning"></i><strong>Risk/Reward:</strong> ${zone.risk_reward_ratio.toFixed(2)}:1</li>
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-success"><i class="fas fa-bullseye me-2"></i>Confluence Score</h6>
                        <div class="d-flex align-items-center mb-3">
                            <div class="progress flex-grow-1 me-3" style="height: 30px;">
                                <div class="progress-bar ${zone.confluence_score >= 80 ? 'bg-success' : 
                                                          zone.confluence_score >= 60 ? 'bg-warning' : 'bg-danger'}" 
                                     style="width: ${zone.confluence_score}%">
                                    <strong class="text-white">${zone.confluence_score.toFixed(0)}%</strong>
                                </div>
                            </div>
                        </div>
                        <small class="text-muted">
                            ${zone.confluence_score >= 80 ? 'Excellent confluence alignment' : 
                              zone.confluence_score >= 60 ? 'Good confluence support' : 'Moderate confluence support'}
                        </small>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-info"><i class="fas fa-layer-group me-2"></i>Why This Target? (${zone.all_confluences ? zone.all_confluences.length : (zone.confluences || []).length} Reasons)</h6>
                        <div class="confluence-explanation mb-3" style="max-height: 150px; overflow-y: auto;">
                            ${(zone.all_confluences || zone.confluences || []).map((conf, idx) => 
                                `<div class="confluence-reason mb-2 p-2 rounded" style="background-color: rgba(23, 162, 184, 0.1); border-left: 3px solid #17a2b8;">
                                    <i class="fas fa-check-circle me-2 text-success"></i>
                                    <strong>${conf}</strong>
                                    <br><small class="text-muted ms-3">${getConfluenceExplanation(conf)}</small>
                                </div>`
                            ).join('')}
                        </div>
                        <div class="confluence-summary p-2 rounded" style="background-color: rgba(40, 167, 69, 0.1); border: 1px solid #28a745;">
                            <small class="text-success fw-bold">
                                <i class="fas fa-lightbulb me-1"></i>
                                ${getTargetSummary(zone)}
                            </small>
                        </div>
                    </div>
                </div>
                
                ${zone.confluence_methods ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6 class="text-secondary"><i class="fas fa-tags me-2"></i>Confluence Categories</h6>
                        <div class="row">
                            ${Object.entries(zone.confluence_methods).filter(([key, methods]) => methods.length > 0).map(([category, methods]) => `
                            <div class="col-md-4 mb-3">
                                <div class="p-2 rounded" style="background-color: rgba(108, 117, 125, 0.1);">
                                    <small class="fw-bold text-capitalize d-block mb-2">
                                        <i class="fas fa-${getCategoryIcon(category)} me-1"></i>
                                        ${category.replace('_', ' ')} (${methods.length})
                                    </small>
                                    <div>
                                        ${methods.map(method => `<span class="badge bg-secondary me-1 mb-1" style="font-size: 0.7em;">${method}</span>`).join('')}
                                    </div>
                                </div>
                            </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="row mt-3">
                    <div class="col-12">
                        <div class="d-flex justify-content-between align-items-center p-2 bg-light rounded">
                            <small class="text-muted">
                                <i class="fas fa-info-circle me-1"></i>
                                Click on table rows above to highlight this target zone
                            </small>
                            <small class="text-muted">
                                Target #${index + 1} of ${targetZones.length}
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    
    detailsDiv.innerHTML = detailsHtml;
}

/**
 * Get explanation for confluence indicators
 */
function getConfluenceExplanation(confluence) {
    const explanations = {
        'RSI Oversold': 'Price likely to bounce from oversold levels (RSI < 30)',
        'RSI Not Oversold': 'RSI indicates room for further downward movement',
        'RSI Overbought': 'Price likely to decline from overbought levels (RSI > 70)',
        'RSI Not Overbought': 'RSI indicates room for further upward movement',
        'MACD Bullish Crossover': 'MACD signal line crossed above main line - bullish momentum',
        'MACD Bearish Crossover': 'MACD signal line crossed below main line - bearish momentum',
        'MACD Bullish Divergence': 'Price makes lower low while MACD makes higher low - reversal signal',
        'MACD Bearish Divergence': 'Price makes higher high while MACD makes lower high - reversal signal',
        'Stochastic Oversold': 'Stochastic oscillator below 20 - oversold, bounce expected',
        'Stochastic Not Oversold': 'Stochastic indicates room for further decline',
        'Stochastic Overbought': 'Stochastic oscillator above 80 - overbought, decline expected',
        'Stochastic Not Overbought': 'Stochastic indicates room for further upward movement',
        'Strong Support': 'Multiple historical price bounces at this level',
        'Major Support': 'Critical support level with high historical significance',
        'Strong Resistance': 'Multiple historical price rejections at this level',
        'Major Resistance': 'Critical resistance level with high historical significance',
        'Fibonacci 23.6%': 'Common retracement level - weak support/resistance',
        'Fibonacci 38.2%': 'Important retracement level - moderate support/resistance',
        'Fibonacci 50.0%': 'Psychological mid-point - strong support/resistance',
        'Fibonacci 61.8%': 'Golden ratio - very strong support/resistance level',
        'Fibonacci 78.6%': 'Deep retracement level - critical support/resistance',
        'Volume Confirmation': 'Trading volume supports the price movement',
        'High Volume': 'Unusually high trading volume confirms price action',
        'Low Volume': 'Low volume suggests weak conviction in price movement',
        'Psychological Level': 'Round number that traders focus on (e.g., $100, $1000)',
        'Previous High': 'Historical resistance at previous peak price',
        'Previous Low': 'Historical support at previous trough price',
        'Moving Average': 'Price interaction with key moving average',
        'Bollinger Band': 'Price at upper or lower Bollinger Band boundary',
        'Trend Line': 'Price at significant trend line support/resistance'
    };
    
    return explanations[confluence] || 'Additional technical analysis confirmation';
}

/**
 * Get target summary explanation
 */
function getTargetSummary(zone) {
    const confluenceCount = zone.all_confluences ? zone.all_confluences.length : (zone.confluences || []).length;
    const probability = (zone.probability * 100).toFixed(0);
    
    if (confluenceCount >= 5) {
        return `Strong Target: ${confluenceCount} technical indicators align at this price level (${probability}% probability)`;
    } else if (confluenceCount >= 3) {
        return `Good Target: ${confluenceCount} indicators confirm this level (${probability}% probability)`;
    } else {
        return `Moderate Target: ${confluenceCount} indicators support this level (${probability}% probability)`;
    }
}

/**
 * Get icon for confluence category
 */
function getCategoryIcon(category) {
    const icons = {
        'fibonacci': 'percentage',
        'support_resistance': 'chart-line',
        'momentum': 'tachometer-alt',
        'pattern': 'shapes',
        'volume': 'chart-bar',
        'harmonic': 'music'
    };
    return icons[category] || 'tag';
}

/**
 * Show detailed confluence modal
 */
function showConfluenceModal(targetIndex) {
    if (!currentAnalysis || !currentAnalysis.target_zones || !currentAnalysis.target_zones[targetIndex]) {
        showNotification('No target zone data available', 'error');
        return;
    }
    
    const zone = currentAnalysis.target_zones[targetIndex];
    const confluences = zone.all_confluences || zone.confluences || [];
    
    // Create modal HTML
    const modalHtml = `
        <div class="modal fade" id="confluenceModal" tabindex="-1" aria-labelledby="confluenceModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title" id="confluenceModalLabel">
                            <i class="fas fa-bullseye me-2"></i>
                            Why Target ${zone.wave_target} at $${zone.price_level.toFixed(4)}?
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <div class="card border-success">
                                    <div class="card-header bg-success text-white">
                                        <i class="fas fa-chart-line me-2"></i>Elliott Wave Basis
                                    </div>
                                    <div class="card-body">
                                        <p><strong>${zone.elliott_basis}</strong></p>
                                        <small class="text-muted">
                                            This target is derived from Elliott Wave theory, specifically targeting the completion 
                                            of ${zone.wave_target} based on Fibonacci relationships and wave structure analysis.
                                        </small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card border-info">
                                    <div class="card-header bg-info text-white">
                                        <i class="fas fa-percentage me-2"></i>Target Statistics
                                    </div>
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between mb-2">
                                            <span>Probability:</span>
                                            <strong class="text-success">${(zone.probability * 100).toFixed(0)}%</strong>
                                        </div>
                                        <div class="d-flex justify-content-between mb-2">
                                            <span>Risk/Reward:</span>
                                            <strong class="text-primary">${zone.risk_reward_ratio.toFixed(2)}:1</strong>
                                        </div>
                                        <div class="d-flex justify-content-between mb-2">
                                            <span>Price Change:</span>
                                            <strong class="${zone.price_change_pct >= 0 ? 'text-success' : 'text-danger'}">
                                                ${zone.price_change_pct >= 0 ? '+' : ''}${zone.price_change_pct.toFixed(2)}%
                                            </strong>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span>Confidence:</span>
                                            <span class="badge ${zone.confidence_level === 'HIGH' ? 'bg-success' : 
                                                               zone.confidence_level === 'MEDIUM' ? 'bg-warning' : 'bg-secondary'}">
                                                ${zone.confidence_level}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card border-warning">
                            <div class="card-header bg-warning text-dark">
                                <i class="fas fa-layer-group me-2"></i>
                                Technical Confluence Analysis (${confluences.length} Supporting Indicators)
                            </div>
                            <div class="card-body">
                                <p class="text-muted mb-3">
                                    <i class="fas fa-lightbulb me-2"></i>
                                    These technical indicators all point to this price level as significant, 
                                    increasing the probability of price reaction at this target.
                                </p>
                                
                                <div class="confluence-explanations">
                                    ${confluences.map((conf, idx) => `
                                        <div class="confluence-item mb-3 p-3 rounded" style="background-color: rgba(0, 123, 255, 0.05); border-left: 4px solid #007bff;">
                                            <div class="d-flex align-items-start">
                                                <div class="me-3">
                                                    <span class="badge bg-primary rounded-circle">${idx + 1}</span>
                                                </div>
                                                <div class="flex-grow-1">
                                                    <h6 class="mb-1 text-primary">${conf}</h6>
                                                    <p class="mb-0 text-muted small">${getConfluenceExplanation(conf)}</p>
                                                </div>
                                                <div class="ms-2">
                                                    <i class="fas fa-check-circle text-success"></i>
                                                </div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                                
                                <div class="mt-4 p-3 bg-success text-white rounded">
                                    <h6 class="mb-2">
                                        <i class="fas fa-star me-2"></i>Confluence Summary
                                    </h6>
                                    <p class="mb-0">${getTargetSummary(zone)}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                            <i class="fas fa-times me-2"></i>Close
                        </button>
                        <button type="button" class="btn btn-primary" onclick="highlightTargetOnChart(${targetIndex})">
                            <i class="fas fa-crosshairs me-2"></i>Highlight on Chart
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('confluenceModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('confluenceModal'));
    modal.show();
}

/**
 * Highlight target on chart
 */
function highlightTargetOnChart(targetIndex) {
    // Close the modal first
    const modal = bootstrap.Modal.getInstance(document.getElementById('confluenceModal'));
    if (modal) modal.hide();
    
    // Scroll to chart
    document.getElementById('chartContainer').scrollIntoView({ behavior: 'smooth' });
    
    // Show notification
    if (currentAnalysis && currentAnalysis.target_zones && currentAnalysis.target_zones[targetIndex]) {
        const zone = currentAnalysis.target_zones[targetIndex];
        showNotification(
            `Highlighted ${zone.wave_target} target at $${zone.price_level.toFixed(4)} on chart`, 
            'success'
        );
    }
}

/**
 * Show detailed confluence information for a specific target zone
 */
function showConfluenceDetails(zone, index) {
    // Highlight the selected confluence detail card
    document.querySelectorAll('.confluence-detail-card').forEach(card => {
        card.classList.remove('border-primary');
    });
    
    const targetCard = document.querySelector(`[data-zone-index="${index}"]`);
    if (targetCard) {
        targetCard.classList.add('border-primary');
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    // Show notification with summary
    const totalConfluences = zone.all_confluences ? zone.all_confluences.length : (zone.confluences || []).length;
    showNotification(
        `Selected ${zone.wave_target}: $${zone.price_level.toFixed(4)} with ${totalConfluences} confluences`, 
        'info'
    );
}

/**
 * Handle keyboard shortcuts
 */
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        performAnalysis();
    }
    
    // Escape to show welcome
    if (e.key === 'Escape') {
        showWelcomeMessage();
    }
});

/**
 * Handle window resize for chart responsiveness
 */
window.addEventListener('resize', function() {
    if (currentAnalysis && !document.getElementById('chartContainer').classList.contains('d-none')) {
        Plotly.Plots.resize('chartDiv');
    }
});

// Expose functions for debugging
window.ElliotWaveBot = {
    performAnalysis,
    exportChart,
    showNotification,
    retryAnalysis,
    changeTimeframeAndRetry
};

/**
 * Retry analysis functionality
 */
function retryAnalysis() {
    console.log('🔄 Retrying analysis...');
    showNotification('Retrying analysis with current settings...', 'info');
    
    // Reset all empty state containers
    resetEmptyStates();
    
    // Trigger new analysis
    performAnalysis();
}

/**
 * Reset all empty state containers and prepare for new analysis
 */
function resetEmptyStates() {
    // Hide all empty state messages
    const emptyStateElements = [
        'noChartDataContainer',
        'noWavesMessage', 
        'noFibonacciMessage',
        'noConfluenceMessage'
    ];
    
    emptyStateElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.add('d-none');
        }
    });
    
    // Reset content containers to default state
    const contentElements = [
        'chartContainer',
        'wavesTableContainer', 
        'fibonacciLevels',
        'confluenceContent'
    ];
    
    contentElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.add('d-none');
        }
    });
    
    // Reset column layouts to default
    const wavesColumn = document.getElementById('wavesResultsColumn');
    const fibonacciColumn = document.getElementById('fibonacciResultsColumn');
    
    if (wavesColumn && fibonacciColumn) {
        wavesColumn.className = 'col-md-6';
        fibonacciColumn.className = 'col-md-6';
        wavesColumn.classList.remove('d-none');
        fibonacciColumn.classList.remove('d-none');
    }
    
    // Hide ASCII table if it exists
    const asciiContainer = document.getElementById('asciiTableContainer');
    if (asciiContainer) {
        asciiContainer.classList.add('d-none');
    }
    
    // Reset badges to 0
    const badges = ['waveCountBadge', 'fibCountBadge', 'confluenceCountBadge'];
    badges.forEach(id => {
        const badge = document.getElementById(id);
        if (badge) {
            badge.textContent = '0';
        }
    });
    
    console.log('✅ Empty states reset for new analysis');
}

/**
 * Change timeframe and automatically retry analysis
 */
function changeTimeframeAndRetry(newTimeframe) {
    console.log(`🔄 Changing timeframe to ${newTimeframe} and retrying...`);
    
    // Update timeframe selection
    const timeframeSelect = document.getElementById('timeframe');
    if (timeframeSelect) {
        timeframeSelect.value = newTimeframe;
        showNotification(`Switched to ${newTimeframe} timeframe and retrying analysis...`, 'info');
        
        // Reset states and retry
        resetEmptyStates();
        performAnalysis();
    } else {
        showNotification('Error: Could not change timeframe', 'error');
    }
}

/**
 * Enhanced error display with retry options
 */
function displayError(message, showRetryOptions = true) {
    // Hide loading state
    document.getElementById('loadingDiv').classList.add('d-none');
    
    // Show error notification
    showNotification(message, 'error');
    
    // Reset empty states to show retry messages
    resetEmptyStates();
    
    // Show no chart data container with retry options if appropriate
    if (showRetryOptions) {
        const noChartContainer = document.getElementById('noChartDataContainer');
        if (noChartContainer) {
            noChartContainer.classList.remove('d-none');
        }
    }
    
    console.error('❌ Analysis error:', message);
}
