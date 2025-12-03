import React, { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResult(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      })

      // Read body ONCE as text and then interpret
      const rawText = await response.text()

      // Try to parse JSON if there is any body
      let parsed = null
      if (rawText) {
        try {
          parsed = JSON.parse(rawText)
        } catch {
          // Not valid JSON, keep parsed = null
        }
      }

      if (!response.ok) {
        let message = 'Failed to process ECG'
        if (parsed && typeof parsed === 'object') {
          message = parsed.detail || JSON.stringify(parsed)
        } else if (rawText) {
          message = rawText
        }
        throw new Error(message)
      }

      if (!parsed) {
        throw new Error(rawText || 'Server returned an empty response')
      }

      const data = parsed
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred during classification')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    // Reset file input
    const fileInput = document.getElementById('file-input')
    if (fileInput) fileInput.value = ''
  }

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🫀 ECG Classification System</h1>
          <p className="subtitle">Upload your ECG PDF or image for AI-powered cardiac analysis</p>
        </header>

        <div className="upload-section">
          <div className="upload-box">
            <input
              id="file-input"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              {file ? '📄 Change File' : '📁 Choose ECG File (PDF or Image)'}
            </label>
            <div className="file-info">
              {file && (
                <div className="file-details">
                  <span className="file-name">📄 {file.name}</span>
                  <span className="file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              )}
            </div>

            <div className="button-group">
              <button
                onClick={handleUpload}
                disabled={!file || uploading}
                className="btn btn-primary"
              >
                {uploading ? 'Processing...' : 'Analyze ECG'}
              </button>
              {file && (
                <button onClick={handleReset} className="btn btn-secondary">
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="error-box">
            <span>⚠️ {error}</span>
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>📊 Classification Results</h2>
            
            <div className="result-card primary">
              <div className="result-label">Predicted Class</div>
              <div className="result-value">{result.predicted_class}</div>
              <div className="result-confidence">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </div>
            </div>

            <div className="probabilities-section">
              <h3>All Class Probabilities</h3>
              <div className="probabilities-grid">
                {Object.entries(result.probabilities).map(([class_name, prob]) => (
                  <div
                    key={class_name}
                    className={`prob-item ${
                      class_name === result.predicted_class ? 'highlight' : ''
                    }`}
                  >
                    <div className="prob-class">{class_name}</div>
                    <div className="prob-bar-container">
                      <div
                        className="prob-bar"
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                    <div className="prob-value">{(prob * 100).toFixed(2)}%</div>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={handleReset} className="btn btn-secondary">
              Analyze Another ECG
            </button>
          </div>
        )}

        <footer className="footer">
          <p>
            Powered by Advanced Hybrid AI Models • 
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              View Source
            </a>
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App

