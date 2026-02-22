import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileImage, FileText, X } from 'lucide-react'

function UploadCard({ onFileSelect, selectedFile, onRemove }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFile(files[0])
    }
  }

  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleFile = (file) => {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf']
    const maxSize = 10 * 1024 * 1024 // 10MB

    if (!allowedTypes.includes(file.type)) {
      alert('Please select a PNG, JPG, or PDF file.')
      return
    }

    if (file.size > maxSize) {
      alert('File size must be less than 10MB.')
      return
    }

    onFileSelect(file)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card max-w-2xl mx-auto"
    >
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${isDragOver
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-primary/50'
          } ${selectedFile ? 'bg-primary/5' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && fileInputRef.current?.click()}
      >
        {selectedFile ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center space-x-3">
              {selectedFile.type === 'application/pdf' ? (
                <FileText className="h-12 w-12 text-primary" />
              ) : (
                <FileImage className="h-12 w-12 text-primary" />
              )}
              <div className="text-left">
                <p className="font-semibold text-text-primary">{selectedFile.name}</p>
                <p className="text-sm text-text-secondary">{formatFileSize(selectedFile.size)}</p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove()
                }}
                className="p-1 hover:bg-red-100 rounded-full transition-colors"
              >
                <X className="h-5 w-5 text-red-500" />
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="h-16 w-16 text-primary mx-auto" />
            <div>
              <h3 className="text-xl font-semibold text-text-primary mb-2">
                Upload Your ECG File
              </h3>
              <p className="text-text-secondary mb-4">
                Drag and drop your ECG file here, or click to browse
              </p>
              <p className="text-sm text-text-secondary">
                Supported formats: PNG, JPG, PDF — 6-Lead Kardia ECG only
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
              className="btn btn-primary"
            >
              Browse Files
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.pdf"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>

      {isUploading && (
        <div className="mt-4">
          <div className="bg-primary/10 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary border-t-transparent"></div>
              <span className="text-text-primary">Analyzing ECG...</span>
            </div>
            <div className="mt-2 bg-border rounded-full h-2">
              <div className="bg-primary h-2 rounded-full animate-pulse w-3/4"></div>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default UploadCard