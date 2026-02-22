import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ResultCard from '../components/ResultCard'
import ECGViewer from '../components/ECGViewer'
import Disclaimer from '../components/Disclaimer'
import { ArrowLeft, Download } from 'lucide-react'

function Results() {
  const [result, setResult] = useState(null)

  useEffect(() => {
    // Load result from localStorage (in real app, this would come from API/state management)
    const storedResult = localStorage.getItem('ecgResult')
    if (storedResult) {
      const parsedResult = JSON.parse(storedResult)
      setResult(parsedResult)
    }
  }, [])

  const handleDownloadReport = () => {
    // Mock download functionality
    alert('Report download feature would be implemented here')
  }

  if (!result) {
    return (
      <div className="max-w-4xl mx-auto text-center py-16">
        <h1 className="text-2xl font-bold text-text-primary mb-4">No Results Found</h1>
        <p className="text-text-secondary mb-8">
          Please upload an ECG file first to see analysis results.
        </p>
        <Link to="/analyze" className="btn btn-primary">
          Upload ECG
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-text-primary mb-2">
            Analysis Results
          </h1>
          <p className="text-text-secondary">
            AI-powered ECG analysis for {result.fileName}
          </p>
        </div>
        <div className="flex space-x-4">
          <button
            onClick={handleDownloadReport}
            className="btn btn-outline inline-flex items-center space-x-2"
          >
            <Download className="h-4 w-4" />
            <span>Download Report</span>
          </button>
          <Link to="/analyze" className="btn btn-secondary inline-flex items-center space-x-2">
            <ArrowLeft className="h-4 w-4" />
            <span>Analyze Another</span>
          </Link>
        </div>
      </motion.div>

      {/* ECG Viewer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <ECGViewer
          src={result.fileUrl}
          fileType={result.fileType}
          fileName={result.fileName}
          fileSize={result.fileSize}
        />
      </motion.div>

      {/* Results */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <ResultCard result={result} />
      </motion.div>

      {/* Analysis Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <h2 className="text-xl font-semibold text-text-primary mb-4">Analysis Summary</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary mb-2">{Math.round(result.confidence * 100)}%</div>
            <div className="text-sm text-text-secondary">AI Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-text-primary mb-2">{result.predicted_class}</div>
            <div className="text-sm text-text-secondary">Primary Diagnosis</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold mb-2 ${result.quality === 'Good' ? 'text-success' : 'text-warning'}`}>
              {result.quality}
            </div>
            <div className="text-sm text-text-secondary">Signal Quality</div>
          </div>
        </div>
      </motion.div>

      <Disclaimer />
    </div>
  )
}

export default Results