import { useState, useEffect, useRef } from "react";
import VideoStream from "./VideoStream";
import circleExclamationPng from '../assets/circle-exclamation.png';
import PropTypes from 'prop-types';
import '../css/UseCamera.css'
import { nanoid } from 'nanoid';

function UseCamera(props) {
    const [videoDevices, setVideoDevices] = useState([]);
    const [vidDeviceId, setVidDeviceId] = useState(null);
    const [recText, setRecText] = useState();
    const [recs, setRecs] = useState();
    const [noCamera, setNoCamera] = useState(false);
    const [cameraErr, setCameraErr] = useState(false);
    const setUseCamera = props.setUseCamera;
    const typewriteIntervalId = useRef(null);

    const startOver = () => {
        setRecs(null);
        setRecText(null);
        setDisplayRecs(null);
        clearInterval(typewriteIntervalId.current);
    };

    useEffect(() => { // Getting video devices
        const getDevices = async () => {
            const devices = await navigator.mediaDevices.enumerateDevices();
            let videoDevices = []
            for (const device of devices) {
                if (device.kind === 'videoinput' && device.deviceId) {
                    videoDevices.push({"deviceId": device.deviceId, "label": device.label});
                }
            }
            
            if (videoDevices.length == 0) setNoCamera(true);
            else setVideoDevices(videoDevices);     
        }
    
        getDevices();
    }, []);


    const [displayRecs, setDisplayRecs] = useState();

    useEffect(() => { // Parsing recText and setting recs
        if (recText) {
            const bulletPoints = recText.split('- **').splice(1);

            setRecs(bulletPoints.map((b) => {
                let pointSplit = b.split("**: ");
                return [pointSplit[0], pointSplit[1].trim()];                    
            }));

            setDisplayRecs(bulletPoints.map(() => ["", ""]));
        }
    }, [recText]);

    const typewriteEffect = (intervalId) => {
        let i = 0;
        let j = 0;
        let c = 0;

        intervalId.current = setInterval(() => {
            if (i < recs.length) {
                if (j < recs[0].length) {
                    if (c < recs[i][j].length) {
                        setDisplayRecs(displayRecs.map((r) => {
                            if (r[j] === displayRecs[i][j]) {
                                r[j] += recs[i][j][c];
                            }
        
                            return r;
                        }));

                        c++;

                    } else {
                        j++;
                        c = 0;
                    }   

                } else {
                    i++;
                    j = 0;
                }
                
            } else {
                return;
            }

        }, 8);
    };

    useEffect(() => {
        if (displayRecs && displayRecs[0][0] === '') {
            typewriteEffect(typewriteIntervalId);
            
            if (typewriteIntervalId) return () => clearInterval(typewriteIntervalId);            
        }

    }, [displayRecs])

    return(
        <div className="use-camera">
            {noCamera && // Screen that appears when client has camera privileges blocked on site
                <div className="no-cam-container">
                    <div>
                        <img className="circle-img" src={circleExclamationPng}/>
                        <h3>No cameras found.</h3>
                        <p>
                            Ensure that permission for website to access camera is granted then reload page.
                        </p>     
                        <button className='btn' onClick={() => setUseCamera(false)}>Home</button>

                    </div>
                </div>
            }

            {videoDevices.length !== 0 && !vidDeviceId && !recText &&
                <div className="device-select">
                    <h3>Select a camera device</h3>

                    {videoDevices.map(
                        (d) => <button className="device-btn" key={d.deviceId} onClick={() => setVidDeviceId(d.deviceId)}>{d.label}</button>
                    )}

                    <button className="device-btn" id="home-btn" onClick={() => setUseCamera(false)}>Home</button>
                </div>
            }

            {vidDeviceId && 
                <VideoStream vidDeviceId={vidDeviceId} setVidDeviceId={setVidDeviceId} setRecText={setRecText} setCameraErr={setCameraErr}/>
            }  

            {cameraErr && // Screen that appears when camera can't pick up video
                <div className="no-cam-container">
                    <div>
                        <img className="circle-img" src={circleExclamationPng}/>
                        <h3>Error with camera.</h3>
                        <p>
                            Ensure camera device is working properly and try again.
                        </p>
                        <button className='btn' onClick={() => setUseCamera(false)}>Home</button>          
                    </div>
                </div>
            }

            {recs && 
                <h2 className='recs-header'>Outfit Recommendations:</h2>
            }

            {displayRecs && 
                <div className="recs-box">
                    <ul className="recs-list">
                        {displayRecs.map((r) => 
                            <li key={nanoid()}>
                                <h3>{r[0]}</h3>
                                <p>{r[1]}</p>
                            </li>                            
                        )}
                    </ul>
                </div>
            }

            {recs && <button className='btn' id="use-camera-btn" onClick={startOver}>Use Camera Again</button>}         
            {recs && <button className='btn' id="use-camera-btn" onClick={() => setUseCamera(false)}>Home</button>}
               
        </div>
    );
}


UseCamera.propTypes = {
    setUseCamera: PropTypes.func
}


export default UseCamera;