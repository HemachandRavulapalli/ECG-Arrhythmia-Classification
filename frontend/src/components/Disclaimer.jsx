import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

function Disclaimer() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="bg-warning/10 border border-warning/20 rounded-lg p-6 mt-8"
    >
      <div className="flex items-start space-x-3">
        <AlertTriangle className="h-6 w-6 text-warning flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-text-primary mb-2">Medical Disclaimer</h3>
          <p className="text-text-secondary text-sm leading-relaxed">
            This system is intended for research and educational purposes only and does not replace professional medical diagnosis.
            Always consult with a qualified healthcare professional for medical advice, diagnosis, and treatment.
            The AI analysis provided here should not be used as a substitute for clinical judgment.
          </p>
        </div>
      </div>
    </motion.div>
  )
}

export default Disclaimer