// File validation utilities for ECG uploads

export const ALLOWED_FILE_TYPES = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'application/pdf': '.pdf'
}

export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB

export const validateFile = (file) => {
  const errors = []

  // Check file type
  if (!Object.keys(ALLOWED_FILE_TYPES).includes(file.type)) {
    errors.push('Invalid file type. Please upload PNG, JPG, or PDF files only.')
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    errors.push('File size too large. Maximum size is 10MB.')
  }

  // Check if it's actually an ECG-related file (basic check)
  if (file.name.toLowerCase().includes('ecg') ||
      file.name.toLowerCase().includes('ekg') ||
      file.name.toLowerCase().includes('cardio')) {
    // This is a bonus check - files with ECG in name are more likely to be correct
  }

  return {
    isValid: errors.length === 0,
    errors
  }
}

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export const getFileType = (file) => {
  if (file.type === 'application/pdf') return 'pdf'
  return 'image'
}