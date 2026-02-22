import HeroSection from '../components/HeroSection'
import SampleECG from '../components/SampleECG'
import Disclaimer from '../components/Disclaimer'

function Home() {
  return (
    <div className="space-y-16">
      <HeroSection />
      <SampleECG />
      <Disclaimer />
    </div>
  )
}

export default Home