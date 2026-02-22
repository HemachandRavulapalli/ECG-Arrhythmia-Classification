import { motion } from 'framer-motion'
import { Heart, CheckCircle, AlertTriangle, Activity } from 'lucide-react'

function ResultCard({ result }) {
  if (!result) return null

  const getQualityIcon = (quality) => {
    return quality === 'Good' ? (
      <CheckCircle className="h-5 w-5 text-success" />
    ) : (
      <AlertTriangle className="h-5 w-5 text-warning" />
    )
  }

  const getQualityColor = (quality) => {
    return quality === 'Good' ? 'text-success bg-success/10' : 'text-warning bg-warning/10'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="grid md:grid-cols-2 gap-6"
    >
      {/* Primary Result */}
      <div className="card bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
        <div className="flex items-center space-x-3 mb-4">
          <Heart className="h-8 w-8 text-primary" />
          <div>
            <h3 className="text-lg font-semibold text-text-primary">Detected Rhythm</h3>
            <p className="text-sm text-text-secondary">AI Analysis Result</p>
          </div>
        </div>
        <div className="space-y-3">
          <div className="text-3xl font-bold text-text-primary">
            {result.predicted_class}
          </div>
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-primary" />
            <span className="text-lg font-semibold text-primary">
              {Math.round(result.confidence * 100)}% Confidence
            </span>
          </div>
        </div>
      </div>

      {/* Signal Quality */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          {getQualityIcon(result.quality)}
          <div>
            <h3 className="text-lg font-semibold text-text-primary">Signal Quality</h3>
            <p className="text-sm text-text-secondary">Recording Assessment</p>
          </div>
        </div>
        <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${getQualityColor(result.quality)}`}>
          {getQualityIcon(result.quality)}
          <span>{result.quality}</span>
        </div>
        <p className="text-sm text-text-secondary mt-3">
          {result.quality === 'Good'
            ? 'Clear signal with minimal noise. High confidence in analysis.'
            : 'Signal contains noise. Results should be interpreted with caution.'
          }
        </p>
      </div>

      {/* Probabilities */}
      <div className="card md:col-span-2">
        <h3 className="text-lg font-semibold text-text-primary mb-4">Class Probabilities</h3>
        <div className="space-y-3">
          {Object.entries(result.probabilities).map(([className, probability]) => (
            <div key={className} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className={`font-medium ${className === result.predicted_class ? 'text-primary' : 'text-text-primary'}`}>
                  {className}
                </span>
                <span className={`text-sm font-semibold ${className === result.predicted_class ? 'text-primary' : 'text-text-secondary'}`}>
                  {probability}%
                </span>
              </div>
              <div className="w-full bg-border rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${probability}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                  className={`h-2 rounded-full ${className === result.predicted_class ? 'bg-primary' : 'bg-secondary/50'}`}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default ResultCard