import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import UploadCard from '../components/UploadCard'
import ECGViewer from '../components/ECGViewer'
import Disclaimer from '../components/Disclaimer'
import { validateFile, getFileType } from '../utils/fileValidation'

function Analyze() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const navigate = useNavigate()

  const handleFileSelect = (file) => {
    const validation = validateFile(file)
    if (!validation.isValid) {
      alert(validation.errors.join('\n'))
      return
    }
    setSelectedFile(file)
  }

  const handleRemoveFile = () => {
    setSelectedFile(null)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minutes timeout

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
      console.log('🚀 Starting upload to:', `${apiUrl}/predict`)
      console.log('📄 File being uploaded:', selectedFile.name, selectedFile.size, selectedFile.type)

      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
        body: formData,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      console.log('📡 Response status:', response.status)
      console.log('📡 Response headers:', Object.fromEntries(response.headers.entries()))

      // Read body ONCE as text and then interpret
      const rawText = await response.text()
      console.log('📄 Raw response text:', rawText)

      // Try to parse JSON if there is any body
      let parsed = null
      if (rawText) {
        try {
          parsed = JSON.parse(rawText)
          console.log('✅ Parsed JSON response:', parsed)
        } catch {
          console.log('❌ Failed to parse JSON, treating as plain text')
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

      const result = parsed

      // Store result in localStorage for the results page
      localStorage.setItem('ecgResult', JSON.stringify({
        ...result,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        fileUrl: URL.createObjectURL(selectedFile),
        fileType: getFileType(selectedFile),
        quality: 'Good'
      }))

      navigate('/results')
    } catch (err) {
      clearTimeout(timeoutId)
      console.error('❌ Upload error:', err)
      if (err.name === 'AbortError') {
        alert('Request timed out. The analysis took too long. Please try again.')
      } else {
        alert(`Analysis failed: ${err.message}`)
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-text-primary mb-4">
          Analyze Your ECG
        </h1>
        <p className="text-text-secondary max-w-2xl mx-auto">
          Upload your 6-lead Kardia ECG file for AI-powered analysis.
          Our system will detect arrhythmias and provide detailed insights.
        </p>
      </motion.div>

      <UploadCard
        onFileSelect={handleFileSelect}
        selectedFile={selectedFile}
        onRemove={handleRemoveFile}
      />

      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <ECGViewer
            src={URL.createObjectURL(selectedFile)}
            fileType={getFileType(selectedFile)}
            fileName={selectedFile.name}
            fileSize={selectedFile.size}
          />
        </motion.div>
      )}

      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-center"
        >
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="btn btn-primary text-lg px-8 py-4"
          >
            {isAnalyzing ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                <span>Analyzing ECG...</span>
              </div>
            ) : (
              'Start Analysis'
            )}
          </button>
        </motion.div>
      )}

      <Disclaimer />
    </div>
  )
}

export default Analyze