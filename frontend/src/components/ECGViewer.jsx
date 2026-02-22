import { useState } from 'react'
import { motion } from 'framer-motion'
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';


pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

function ECGViewer({ src, fileType, fileName, fileSize }) {
  const [zoom, setZoom] = useState(1)
  const [rotation, setRotation] = useState(0)
  const [numPages, setNumPages] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5))
  const handleReset = () => {
    setZoom(1)
    setRotation(0)
  }

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages)
    setPageNumber(1)
  }

  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1))
  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages))

  if (!src) {
    return (
      <div className="bg-border/50 rounded-lg p-8 text-center">
        <p className="text-text-secondary">No ECG file available</p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text-primary">ECG Viewer</h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleZoomOut}
            className="p-2 hover:bg-border rounded-lg transition-colors"
            disabled={zoom <= 0.5}
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="text-sm text-text-secondary min-w-[60px] text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-2 hover:bg-border rounded-lg transition-colors"
            disabled={zoom >= 3}
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 hover:bg-border rounded-lg transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="bg-border/20 rounded-lg p-4 overflow-auto max-h-96">
        <div
          className="flex justify-center"
          style={{
            transform: `scale(${zoom}) rotate(${rotation}deg)`,
            transformOrigin: 'center',
            transition: 'transform 0.2s ease'
          }}
        >
          {fileType === 'pdf' ? (
            <div className="bg-white border border-border rounded-lg p-4 shadow-sm">
              <Document
                file={src}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={<div className="text-center py-8 text-text-secondary">Loading PDF...</div>}
                error={<div className="text-center py-8 text-red-500">Failed to load PDF.</div>}
              >
                <Page pageNumber={pageNumber} width={350 * zoom} />
              </Document>
              <div className="flex items-center justify-center mt-2 space-x-2">
                <button onClick={goToPrevPage} disabled={pageNumber <= 1} className="btn btn-outline px-2 py-1">Prev</button>
                <span className="text-xs text-text-secondary">Page {pageNumber} of {numPages}</span>
                <button onClick={goToNextPage} disabled={pageNumber >= numPages} className="btn btn-outline px-2 py-1">Next</button>
              </div>
              <p className="text-xs text-text-secondary mt-2">File: {fileName}</p>
            </div>
          ) : (
            <img
              src={src}
              alt="ECG Report"
              className="max-w-full h-auto border border-border rounded-lg shadow-sm"
              style={{ maxHeight: '400px' }}
            />
          )}
        </div>
      </div>

      <div className="mt-4 text-sm text-text-secondary">
        <p>Zoom: {Math.round(zoom * 100)}% | File: {fileName} | Size: {(fileSize / 1024 / 1024).toFixed(2)} MB</p>
      </div>
    </motion.div>
  )
}

export default ECGViewer