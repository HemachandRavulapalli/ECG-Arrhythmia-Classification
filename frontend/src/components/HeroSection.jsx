import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity, Upload, Play } from 'lucide-react'

function HeroSection() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="text-center py-16"
    >
      <div className="max-w-4xl mx-auto">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-4xl md:text-6xl font-bold text-text-primary mb-6"
        >
          AI-Assisted 6-Lead ECG Interpretation
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="text-xl text-text-secondary mb-12 max-w-2xl mx-auto"
        >
          Upload or explore a sample Kardia 6-Lead ECG and receive AI-based rhythm analysis.
        </motion.p>

        {/* ECG Waveform Illustration */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mb-12"
        >
          <div className="bg-card rounded-xl p-8 shadow-sm border border-border inline-block">
            <Activity className="h-24 w-24 text-primary mx-auto mb-4" />
            <div className="flex justify-center space-x-2">
              <div className="w-16 h-8 bg-primary/20 rounded animate-pulse"></div>
              <div className="w-12 h-6 bg-secondary/20 rounded animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-20 h-10 bg-primary/20 rounded animate-pulse" style={{ animationDelay: '0.4s' }}></div>
              <div className="w-14 h-7 bg-secondary/20 rounded animate-pulse" style={{ animationDelay: '0.6s' }}></div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Link to="/analyze" className="btn btn-primary inline-flex items-center space-x-2">
            <Upload className="h-5 w-5" />
            <span>Upload Your ECG</span>
          </Link>
          <Link to="/results" className="btn btn-outline inline-flex items-center space-x-2">
            <Play className="h-5 w-5" />
            <span>Try Sample ECG</span>
          </Link>
        </motion.div>
      </div>
    </motion.div>
  )
}

export default HeroSection