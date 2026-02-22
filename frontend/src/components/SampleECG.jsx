import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FileImage, FileText, Play } from 'lucide-react'

function SampleECG() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="card max-w-4xl mx-auto"
    >
      <div className="text-center mb-8">
        <h2 className="text-2xl font-semibold text-text-primary mb-4">
          Explore with Sample ECG
        </h2>
        <p className="text-text-secondary">
          Use this sample ECG to understand how the AI analyzes heart rhythms.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Sample Image */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2 mb-4">
            <FileImage className="h-5 w-5 text-primary" />
            <span className="font-medium text-text-primary">Sample ECG Image</span>
          </div>
          <div className="bg-border/50 rounded-lg p-8 text-center">
            <img
              src="/sample_ecg.jpg"
              alt="Sample ECG"
              className="h-40 mx-auto mb-4 border border-border rounded-lg shadow"
              style={{ objectFit: 'contain' }}
            />
            <p className="text-text-secondary">6-Lead ECG Report</p>
            <p className="text-sm text-text-secondary mt-2">Sample Kardia Report</p>
            <a
              href="/sample_ecg.jpg"
              download
              className="btn btn-secondary mt-2"
            >
              Download Image
            </a>
          </div>
        </div>

        {/* Sample PDF */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2 mb-4">
            <FileText className="h-5 w-5 text-primary" />
            <span className="font-medium text-text-primary">Sample ECG PDF</span>
          </div>
          <div className="bg-border/50 rounded-lg p-8 text-center">
            <iframe
              src="/sample_ecg.pdf"
              title="Sample ECG PDF"
              className="w-full h-40 border border-border rounded-lg mb-4"
            />
            <p className="text-text-secondary">6-Lead ECG Report</p>
            <p className="text-sm text-text-secondary mt-2">Digital PDF Report</p>
            <a
              href="/sample_ecg.pdf"
              download
              className="btn btn-secondary mt-2"
            >
              Download PDF
            </a>
          </div>
        </div>
      </div>

      <div className="text-center mt-8">
        <Link to="/results" className="btn btn-outline inline-flex items-center space-x-2">
          <Play className="h-5 w-5" />
          <span>Analyze Sample ECG</span>
        </Link>
      </div>
    </motion.div>
  )
}

export default SampleECG