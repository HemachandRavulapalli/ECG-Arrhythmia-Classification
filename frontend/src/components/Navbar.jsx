import { Link } from 'react-router-dom'
import { Heart } from 'lucide-react'

function Navbar() {
  return (
    <nav className="bg-card border-b border-border shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Heart className="h-8 w-8 text-primary" />
            <span className="text-xl font-semibold text-text-primary">ECG Insight AI</span>
          </Link>
          <div className="hidden md:flex space-x-8">
            <Link to="/" className="text-text-secondary hover:text-primary transition-colors">
              Home
            </Link>
            <Link to="/analyze" className="text-text-secondary hover:text-primary transition-colors">
              Analyze
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar