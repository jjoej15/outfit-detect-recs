import { useState, useEffect } from 'react'
import VideoStream from './components/VideoStream'
import './App.css'

function App() {
  const [videoDevices, setVideoDevices] = useState([]);
  const [vidDeviceId, setVidDeviceId] = useState(null);
  const [recText, setRecText] = useState();

  useEffect(() => { // Getting video devices
    const getDevices = async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      let videoDevices = []
      for (const device of devices) {
          if (device.kind === 'videoinput' && device.deviceId) {
              videoDevices.push({"deviceId": device.deviceId, "label": device.label});
          }
      }

      setVideoDevices(videoDevices);       
    }

    getDevices();
  }, []);

  return (
    <div className='app'>
      {videoDevices && videoDevices.map((d) => <button type='button' key={d.deviceId} onClick={() => setVidDeviceId(d.deviceId)}>{d.label}</button>)}
      {vidDeviceId && <VideoStream vidDeviceId={vidDeviceId} setVidDeviceId={setVidDeviceId} setRecText={setRecText}/>}  
      {recText && <p>{recText}</p>}
    </div>

  );
}

export default App
