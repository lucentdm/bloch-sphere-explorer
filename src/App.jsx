import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

const GATES = {
  I: [[{r:1,i:0},{r:0,i:0}],[{r:0,i:0},{r:1,i:0}]],
  X: [[{r:0,i:0},{r:1,i:0}],[{r:1,i:0},{r:0,i:0}]],
  Y: [[{r:0,i:0},{r:0,i:-1}],[{r:0,i:1},{r:0,i:0}]],
  Z: [[{r:1,i:0},{r:0,i:0}],[{r:0,i:0},{r:-1,i:0}]],
  H: [[{r:1/Math.SQRT2,i:0},{r:1/Math.SQRT2,i:0}],[{r:1/Math.SQRT2,i:0},{r:-1/Math.SQRT2,i:0}]],
  S: [[{r:1,i:0},{r:0,i:0}],[{r:0,i:0},{r:0,i:1}]],
  T: [[{r:1,i:0},{r:0,i:0}],[{r:0,i:0},{r:Math.cos(Math.PI/4),i:Math.sin(Math.PI/4)}]],
};

function mulMV(m, v) {
  return [
    {r: m[0][0].r*v[0].r - m[0][0].i*v[0].i + m[0][1].r*v[1].r - m[0][1].i*v[1].i,
     i: m[0][0].r*v[0].i + m[0][0].i*v[0].r + m[0][1].r*v[1].i + m[0][1].i*v[1].r},
    {r: m[1][0].r*v[0].r - m[1][0].i*v[0].i + m[1][1].r*v[1].r - m[1][1].i*v[1].i,
     i: m[1][0].r*v[0].i + m[1][0].i*v[0].r + m[1][1].r*v[1].i + m[1][1].i*v[1].r},
  ];
}

function stateToBloch(alpha, beta) {
  const norm = Math.sqrt(alpha.r**2+alpha.i**2+beta.r**2+beta.i**2);
  if (norm < 1e-10) return {x:0,y:0,z:1};
  const a = {r:alpha.r/norm, i:alpha.i/norm};
  const b = {r:beta.r/norm, i:beta.i/norm};
  return {
    x: 2*(a.r*b.r + a.i*b.i),
    y: 2*(a.i*b.r - a.r*b.i),
    z: a.r**2 + a.i**2 - b.r**2 - b.i**2,
  };
}

function slerpBloch(v1, v2, t) {
  const dot = Math.min(1, Math.max(-1,
    v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
  ));

  // Special case to deal with antipodal singularity for 180 degree rotations
  // Same direction
  if (dot > 0.999999) return v2;

  // Opposite directions: choose a stable great-circle path
  if (dot < -0.999999) {
    let ortho;

    // Pick an axis that isn't parallel to v1
    if (Math.abs(v1.x) < 0.9) {
      ortho = {
        x: 0,
        y: v1.z,
        z: -v1.y
      };
    } else {
      ortho = {
        x: -v1.z,
        y: 0,
        z: v1.x
      };
    }

    const len = Math.sqrt(
      ortho.x**2 + ortho.y**2 + ortho.z**2
    );

    ortho.x /= len;
    ortho.y /= len;
    ortho.z /= len;

    const angle = Math.PI * t;

    return {
      x: Math.cos(angle)*v1.x + Math.sin(angle)*ortho.x,
      y: Math.cos(angle)*v1.y + Math.sin(angle)*ortho.y,
      z: Math.cos(angle)*v1.z + Math.sin(angle)*ortho.z,
    };
  }

  // Normal SLERP
  const omega = Math.acos(dot);
  const s = Math.sin(omega);

  const f1 = Math.sin((1-t)*omega) / s;
  const f2 = Math.sin(t*omega) / s;

  return {
    x: f1*v1.x + f2*v2.x,
    y: f1*v1.y + f2*v2.y,
    z: f1*v1.z + f2*v2.z
  };
}

function parseMatrix(rows) {
  try {
    return rows.map(row => row.map(cell => {
      const s = cell.trim();
      if (!s) return null;
      let r = 0, im = 0;
      if (s.includes('i')) {
        const iMatch = s.match(/([+-]?\d*\.?\d*)i/);
        const rPart = s.replace(/([+-]?\d*\.?\d*)i/, '').trim();
        r = rPart ? (parseFloat(rPart)||0) : 0;
        im = iMatch ? (iMatch[1]===''||iMatch[1]==='+' ? 1 : iMatch[1]==='-' ? -1 : parseFloat(iMatch[1])||1) : 0;
      } else {
        r = parseFloat(s)||0;
      }
      return {r, i:im};
    }));
  } catch { return null; }
}

const defaultMatrixStr = [["1","0"],["0","1"]];

// --- HTML Label overlay ---
function AxisLabels({ camera, groupRef, renderer }) {
  const labels = [
    { pos: new THREE.Vector3(0, 1.45, 0),  text: "|0⟩",  color: "#00ff88" },
    { pos: new THREE.Vector3(0, -1.55, 0), text: "|1⟩",  color: "#ff4488" },
    { pos: new THREE.Vector3(1.45, 0, 0),  text: "|+⟩",  color: "#00f0ff" },
    { pos: new THREE.Vector3(-1.55, 0, 0), text: "|−⟩",  color: "#00f0ff" },
    { pos: new THREE.Vector3(0, 0, 1.45),  text: "|i⟩",  color: "#cc44ff" },
    { pos: new THREE.Vector3(0, 0, -1.55), text: "|−i⟩", color: "#cc44ff" },
  ];
  const [positions, setPositions] = useState([]);

  useEffect(() => {
    let raf;
    const update = () => {
      raf = requestAnimationFrame(update);
      if (!camera || !groupRef.current || !renderer) return;
      const canvas = renderer.domElement;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const pts = labels.map(({ pos, text, color }) => {
        const worldPos = pos.clone().applyMatrix4(groupRef.current.matrixWorld);
        const ndc = worldPos.clone().project(camera);
        const x = (ndc.x * 0.5 + 0.5) * w;
        const y = (1 - (ndc.y * 0.5 + 0.5)) * h;
        const behind = ndc.z > 1;
        return { x, y, text, color, behind };
      });
      setPositions(pts);
    };
    update();
    return () => cancelAnimationFrame(raf);
  }, [camera, groupRef, renderer]);

  return (
    <div style={{ position:'absolute', inset:0, pointerEvents:'none' }}>
      {positions.map((p, i) => (
        !p.behind && (
          <div key={i} style={{
            position:'absolute',
            left: p.x, top: p.y,
            transform: 'translate(-50%, -50%)',
            color: p.color,
            fontSize: '1.25em',
            fontWeight: 'bold',
            fontFamily: "'Courier New', monospace",
            textShadow: `0 0 8px ${p.color}, 0 0 16px ${p.color}`,
            background: 'rgba(5,5,20,0.9)',
            padding: '4px 8px',
            borderRadius: 4,
            border: `1px solid ${p.color}44`,
            letterSpacing: '0.05em',
            whiteSpace: 'nowrap',
          }}>{p.text}</div>
        )
      ))}
    </div>
  );
}

export default function BlochSphere() {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const arrowRef = useRef(null);
  const sphereGroupRef = useRef(null);
  const blochPosRef = useRef({x:0,y:0,z:1});
  const isDragging = useRef(false);
  const prevMouse = useRef({x:0,y:0});
  const zoomRef = useRef(4.5); // camera distance

  const [ops, setOps] = useState([{name:"H", matrix: GATES.H.map(r=>r.map(c=>({...c})))}]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [matrixInputs, setMatrixInputs] = useState(defaultMatrixStr);
  const [customName, setCustomName] = useState("");
  const [stateInfo, setStateInfo] = useState({alpha:{r:1,i:0},beta:{r:0,i:0}});
  const [rendererReady, setRendererReady] = useState(false);

  const playRef = useRef(false);
  const stepRef = useRef(0);
  const opsRef = useRef(ops);
  useEffect(()=>{opsRef.current=ops;},[ops]);
  useEffect(()=>{stepRef.current=step;},[step]);
  useEffect(()=>{playRef.current=playing;},[playing]);

  const computeStates = useCallback((operations) => {
    let state = [{r:1,i:0},{r:0,i:0}];
    const states = [state];
    for (const op of operations) { state = mulMV(op.matrix, state); states.push(state); }
    return states;
  }, []);
  const statesRef = useRef(computeStates(ops));
  useEffect(() => { statesRef.current = computeStates(ops); }, [ops, computeStates]);

  // Three.js setup
  useEffect(() => {
    const w = mountRef.current.clientWidth;
    const h = mountRef.current.clientHeight;

    const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 100);
    camera.position.set(2.8, 2.0, 2.8);
    camera.lookAt(0,0,0);
    zoomRef.current = camera.position.length();
    cameraRef.current = camera;

    const group = new THREE.Group();
    sphereGroupRef.current = group;
    scene.add(group);

    // Sphere
    const sphereMat = new THREE.MeshPhongMaterial({
      color: 0x0a2a33,
      emissive: 0x06181c,
      // color: 0x14144a,
      // emissive: 0x080820,
      transparent: true,
      opacity: 0.65,
      specular: 0x66aaff,
      shininess: 100
    });
    // Old, darker coloring
    //const sphereMat = new THREE.MeshPhongMaterial({color:0x0a0a2a,transparent:true,opacity:0.5,emissive:0x0a0a2a,specular:0x4488ff,shininess:80});
    group.add(new THREE.Mesh(new THREE.SphereGeometry(1,48,48), sphereMat));

    // Adding another light to brighten it up a bit
    scene.add(new THREE.AmbientLight(0x334466, 3));
    const fill = new THREE.DirectionalLight(0x6688ff, 1.2);
    fill.position.set(2, 3, 4);
    scene.add(fill);

    // Wireframe circles
    [[0x00f0ff,false,true],[0xcc44ff,true,false],[0x00f0ff,false,false]].forEach(([color,xz,xy],idx)=>{
      const geo = new THREE.BufferGeometry();
      const pts = [];
      for (let i=0;i<=128;i++) {
        const a=(i/128)*Math.PI*2;
        if (idx===0) pts.push(Math.cos(a),Math.sin(a),0);
        else if (idx===1) pts.push(Math.cos(a),0,Math.sin(a));
        else pts.push(0,Math.cos(a),Math.sin(a));
      }
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pts,3));
      group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({color,transparent:true,opacity:0.6})));
    });

    // Axis lines (thin dashed-look via segments)
    const axisLines = [
      {from:[0,-1.2,0],to:[0,1.2,0],color:0x00ff88},
      {from:[-1.2,0,0],to:[1.2,0,0],color:0x00f0ff},
      {from:[0,0,-1.2],to:[0,0,1.2],color:0xcc44ff},
    ];
    axisLines.forEach(({from,to,color})=>{
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...from),new THREE.Vector3(...to)]);
      group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({color,transparent:true,opacity:0.85})));
    });

    // Bloch vector arrow
    const arrowGroup = new THREE.Group();
    group.add(arrowGroup);
    arrowRef.current = arrowGroup;

    const shaftMat = new THREE.MeshPhongMaterial({color:0xffee00,emissive:0xffaa00,emissiveIntensity:1.5});
    const shaftGeo = new THREE.CylinderGeometry(0.04,0.04,0.85,16); shaftGeo.translate(0,0.425,0);
    arrowGroup.add(new THREE.Mesh(shaftGeo, shaftMat));
    const headGeo = new THREE.ConeGeometry(0.09,0.18,16); headGeo.translate(0,0.925,0);
    arrowGroup.add(new THREE.Mesh(headGeo, shaftMat));
    const glowGeo = new THREE.SphereGeometry(0.07,12,12); glowGeo.translate(0,1,0);
    arrowGroup.add(new THREE.Mesh(glowGeo, new THREE.MeshPhongMaterial({color:0xffff00,emissive:0xffff00,emissiveIntensity:3,transparent:true,opacity:0.8})));

    // Lighting
    scene.add(new THREE.AmbientLight(0x222244,2));
    const pt = new THREE.PointLight(0x4488ff,2,10); pt.position.set(3,3,3); scene.add(pt);
    const pt2 = new THREE.PointLight(0xff44cc,1.5,10); pt2.position.set(-2,-2,2); scene.add(pt2);

    // Stars
    const starPts = [];
    for (let i=0;i<500;i++) starPts.push((Math.random()-0.5)*40,(Math.random()-0.5)*40,(Math.random()-0.5)*40);
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPts,3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({color:0xffffff,size:0.05,transparent:true,opacity:0.6})));

    // Grid
    const grid = new THREE.GridHelper(6,12,0x1a0a3a,0x1a0a3a); grid.position.y=-1.5; scene.add(grid);

    let raf;
    const animate = () => { raf=requestAnimationFrame(animate); renderer.render(scene,camera); };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      const w2=mountRef.current.clientWidth, h2=mountRef.current.clientHeight;
      renderer.setSize(w2,h2); camera.aspect=w2/h2; camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);
    setRendererReady(true);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      if (mountRef.current && renderer.domElement.parentNode===mountRef.current)
        mountRef.current.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // Mouse drag + scroll zoom
  useEffect(() => {
    const el = mountRef.current;
    const onDown = e => { isDragging.current=true; prevMouse.current={x:e.clientX,y:e.clientY}; };
    const onUp = () => { isDragging.current=false; };
    const onMove = e => {
      if (!isDragging.current||!sphereGroupRef.current) return;
      const dx=e.clientX-prevMouse.current.x, dy=e.clientY-prevMouse.current.y;
      prevMouse.current={x:e.clientX,y:e.clientY};
      sphereGroupRef.current.rotation.y += dx*0.01;
      sphereGroupRef.current.rotation.x += dy*0.01;
    };
    const onWheel = e => {
      e.preventDefault();
      const cam = cameraRef.current;
      if (!cam) return;
      const dist = cam.position.length();
      const newDist = Math.min(9, Math.max(1.5, dist + e.deltaY * 0.005));
      cam.position.setLength(newDist);
      zoomRef.current = newDist;
    };
    el.addEventListener('mousedown', onDown);
    el.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    return () => {
      el.removeEventListener('mousedown', onDown);
      el.removeEventListener('wheel', onWheel);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('mousemove', onMove);
    };
  }, []);

  // Zoom buttons
  const zoom = useCallback((dir) => {
    const cam = cameraRef.current;
    if (!cam) return;
    const dist = cam.position.length();
    const newDist = Math.min(9, Math.max(1.5, dist + dir * 0.4));
    cam.position.setLength(newDist);
    zoomRef.current = newDist;
  }, []);

  const updateArrow = useCallback((bloch) => {
    if (!arrowRef.current) return;
    const v = new THREE.Vector3(bloch.x, bloch.z, -bloch.y).normalize();
    arrowRef.current.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0), v));
    blochPosRef.current = bloch;
  }, []);

  const animateTo = useCallback((fromBloch, toBloch, onDone) => {
    let start = null;
    const dur = 800;
    const tick = ts => {
      if (!start) start = ts;
      const t = Math.min((ts-start)/dur, 1);
      const ease = t<0.5 ? 2*t*t : -1+(4-2*t)*t;
      updateArrow(slerpBloch(fromBloch, toBloch, ease));
      if (t < 1) requestAnimationFrame(tick);
      else { updateArrow(toBloch); onDone && onDone(); }
    };
    requestAnimationFrame(tick);
  }, [updateArrow]);

  useEffect(() => {
    const states = statesRef.current;
    const bloch = stateToBloch(states[step][0], states[step][1]);
    const prev = step > 0 ? stateToBloch(states[step-1][0], states[step-1][1]) : blochPosRef.current;
    animateTo(prev, bloch, ()=>{});
    setStateInfo({alpha:states[step][0], beta:states[step][1]});
  }, [step, animateTo]);

  useEffect(() => {
    if (!playing) return;
    const total = opsRef.current.length;
    let cur = stepRef.current;
    const advance = () => {
      if (!playRef.current) return;
      if (cur >= total) { setPlaying(false); return; }
      cur++; setStep(cur);
      setTimeout(advance, 1000);
    };
    advance();
  }, [playing]);

  const fmtC = ({r,i}) => {
    const rs = Math.abs(r)<0.001?'':r.toFixed(3);
    const is = Math.abs(i)<0.001?'':(i>=0&&rs?'+':'')+i.toFixed(3)+'i';
    return (rs+is)||'0';
  };

  const addGate = name => setOps(prev => [...prev, {name, matrix: GATES[name].map(r=>r.map(c=>({...c})))}]);
  const addCustom = () => {
    const m = parseMatrix(matrixInputs);
    if (!m || m.some(r=>r.some(c=>c===null))) return;
    setOps(prev => [...prev, {name: customName||'U', matrix:m}]);
  };
  const removeOp = idx => { setOps(prev=>prev.filter((_,i)=>i!==idx)); setStep(s => Math.max(0, Math.min(s, ops.length - 1))); };

  const theta = Math.acos(Math.max(-1,Math.min(1, stateInfo.alpha.r**2+stateInfo.alpha.i**2-stateInfo.beta.r**2-stateInfo.beta.i**2)));
  const phi = Math.atan2(stateInfo.beta.i*stateInfo.alpha.r-stateInfo.beta.r*stateInfo.alpha.i,
                         stateInfo.beta.r*stateInfo.alpha.r+stateInfo.beta.i*stateInfo.alpha.i)*2;

  return (
    <div style={{background:'#050510',minHeight:'100vh',display:'flex',flexDirection:'column',fontFamily:"'Courier New',monospace",color:'#00f0ff',overflow:'hidden'}}>
      <div style={{textAlign:'center',padding:'10px 0 4px',background:'linear-gradient(180deg,#0a0a2a,transparent)'}}>
        <div style={{fontSize:'1.3em',fontWeight:'bold',letterSpacing:'0.18em',color:'#cc44ff',textShadow:'0 0 16px #cc44ff,0 0 32px #8800cc'}}>BLOCH SPHERE EXPLORER</div>
        <div style={{fontSize:'0.7em',color:'#4488ff',letterSpacing:'0.12em'}}>QUANTUM STATE VISUALIZER</div>
      </div>

      <div style={{display:'flex',flex:1,gap:'10px',padding:'6px 10px',minHeight:0}}>
        {/* 3D View */}
        <div style={{flex:'1 1 0',minWidth:0,position:'relative',borderRadius:'10px',border:'1px solid #1a0a3a',overflow:'hidden',boxShadow:'0 0 30px #0a0a4a'}}>
          <div ref={mountRef} style={{width:'100%',height:'100%',minHeight:'340px',cursor:'grab'}}/>

          {/* Axis labels overlay */}
          {rendererReady && cameraRef.current && rendererRef.current && (
            <AxisLabels camera={cameraRef.current} groupRef={sphereGroupRef} renderer={rendererRef.current}/>
          )}

          {/* Zoom buttons */}
          <div style={{position:'absolute',top:8,right:8,display:'flex',flexDirection:'column',gap:4}}>
            {[['＋', -1],['－', 1]].map(([label, dir])=>(
              <button key={label} onClick={()=>zoom(dir)} style={{
                width:32,height:32,background:'rgba(10,10,40,0.85)',border:'1px solid #4400aa',
                color:'#cc44ff',fontSize:'1.1em',borderRadius:6,cursor:'pointer',
                fontFamily:'inherit',textShadow:'0 0 8px #cc44ff',
                boxShadow:'0 0 10px rgba(150,0,255,0.3)',display:'flex',alignItems:'center',justifyContent:'center',
              }}>{label}</button>
            ))}
          </div>

          <div style={{position:'absolute',top:8,left:8,fontSize:'0.62em',color:'#4466aa',pointerEvents:'none'}}>drag to rotate · scroll or +/− to zoom</div>

          {/* State display */}
          <div style={{position:'absolute',bottom:8,left:8,background:'rgba(5,5,20,0.85)',border:'1px solid #1a0a5a',borderRadius:6,padding:'8px 12px',fontSize:'0.9em',lineHeight:1.7}}>
            <div style={{color:'#cc44ff',fontWeight:'bold',letterSpacing:'0.1em',marginBottom:2}}>CURRENT STATE</div>
            <div>|ψ⟩ = <span style={{color:'#00f0ff'}}>{fmtC(stateInfo.alpha)}</span>|0⟩ + <span style={{color:'#ff44cc'}}>{fmtC(stateInfo.beta)}</span>|1⟩</div>
            <div>θ = <span style={{color:'#ffee00'}}>{(theta*180/Math.PI).toFixed(1)}°</span>  φ = <span style={{color:'#ffee00'}}>{(phi*180/Math.PI).toFixed(1)}°</span></div>
            <div style={{color:'#00ff88',fontSize:'0.9em'}}>Step {step} of {ops.length}</div>
          </div>
        </div>

        {/* Control panel */}
        <div style={{width:'240px',display:'flex',flexDirection:'column',gap:'8px',overflowY:'auto'}}>
          {/* Preset gates */}
          <div style={{background:'rgba(10,10,40,0.9)',border:'1px solid #1a0a5a',borderRadius:8,padding:'10px'}}>
            <div style={{color:'#cc44ff',fontWeight:'bold',fontSize:'0.75em',letterSpacing:'0.12em',marginBottom:8}}>⚡ PRESET GATES</div>
            <div style={{display:'flex',flexWrap:'wrap',gap:5}}>
              {Object.keys(GATES).filter(g=>g!=='I').map(g=>(
                <button key={g} onClick={()=>addGate(g)} style={{
                  background:'linear-gradient(135deg,#0a0a3a,#1a0a4a)',border:'1px solid #4400aa',
                  color:'#00f0ff',padding:'4px 10px',borderRadius:5,cursor:'pointer',fontSize:'0.8em',
                  fontFamily:'inherit',textShadow:'0 0 8px #00f0ff',boxShadow:'0 0 8px rgba(0,150,255,0.2)',
                }}>{g}</button>
              ))}
            </div>
          </div>

          {/* Custom matrix */}
          <div style={{background:'rgba(10,10,40,0.9)',border:'1px solid #1a0a5a',borderRadius:8,padding:'10px'}}>
            <div style={{color:'#cc44ff',fontWeight:'bold',fontSize:'0.75em',letterSpacing:'0.12em',marginBottom:8}}>🔢 CUSTOM GATE</div>
            <div style={{fontSize:'0.65em',color:'#4488ff',marginBottom:6}}>e.g. 0.707, 1i, -0.5+0.5i</div>
            {[0,1].map(r=>(
              <div key={r} style={{display:'flex',gap:4,marginBottom:4}}>
                {[0,1].map(c=>(
                  <input key={c} value={matrixInputs[r][c]}
                    onChange={e=>{const m=matrixInputs.map(rr=>[...rr]);m[r][c]=e.target.value;setMatrixInputs(m);}}
                    style={{width:'100%',background:'#05051a',border:'1px solid #2200aa',color:'#00f0ff',
                      padding:'4px',borderRadius:4,fontSize:'0.75em',fontFamily:'inherit',textAlign:'center'}}/>
                ))}
              </div>
            ))}
            <div style={{display:'flex',gap:5,marginTop:5}}>
              <input value={customName} onChange={e=>setCustomName(e.target.value)} placeholder="Name"
                style={{width:'60px',background:'#05051a',border:'1px solid #2200aa',color:'#cc44ff',
                  padding:'4px',borderRadius:4,fontSize:'0.75em',fontFamily:'inherit'}}/>
              <button onClick={addCustom} style={{flex:1,background:'linear-gradient(135deg,#1a004a,#2200aa)',
                border:'1px solid #6600ff',color:'#cc44ff',padding:'4px',borderRadius:4,
                cursor:'pointer',fontSize:'0.75em',fontFamily:'inherit',fontWeight:'bold',
                textShadow:'0 0 8px #cc44ff'}}>+ ADD</button>
            </div>
          </div>

          {/* Sequence */}
          <div style={{background:'rgba(10,10,40,0.9)',border:'1px solid #1a0a5a',borderRadius:8,padding:'10px',flex:1}}>
            <div style={{color:'#cc44ff',fontWeight:'bold',fontSize:'0.75em',letterSpacing:'0.12em',marginBottom:8}}>📋 SEQUENCE</div>
            <div style={{fontSize:'0.65em',color:'#4488ff',marginBottom:6}}>Start: |0⟩</div>
            <div style={{maxHeight:'130px',overflowY:'auto'}}>
              {ops.map((op,i)=>(
                <div key={i} onClick={()=>setStep(i+1)} style={{
                  display:'flex',alignItems:'center',justifyContent:'space-between',
                  padding:'3px 6px',marginBottom:3,borderRadius:4,cursor:'pointer',
                  background: step===i+1?'linear-gradient(135deg,#1a004a,#2200aa)':'rgba(20,0,60,0.5)',
                  border: step===i+1?'1px solid #6600ff':'1px solid #1a0a3a',
                  color: step===i+1?'#cc44ff':'#8866ff',
                  textShadow: step===i+1?'0 0 8px #cc44ff':'none',
                  fontSize:'0.85em',
                }}>
                  <span>{i+1}. {op.name}</span>
                  <span onClick={e=>{e.stopPropagation();removeOp(i);}} style={{color:'#ff4466',cursor:'pointer',fontSize:'1.1em',padding:'0 2px'}}>×</span>
                </div>
              ))}
            </div>
          </div>

          {/* Playback */}
          <div style={{background:'rgba(10,10,40,0.9)',border:'1px solid #1a0a5a',borderRadius:8,padding:'10px'}}>
            <div style={{color:'#cc44ff',fontWeight:'bold',fontSize:'0.75em',letterSpacing:'0.12em',marginBottom:8}}>▶ PLAYBACK</div>
            <input type="range" min={0} max={ops.length} value={step}
              onChange={e=>{setPlaying(false);setStep(Number(e.target.value));}}
              style={{width:'100%',accentColor:'#cc44ff',marginBottom:8}}/>
            <div style={{display:'flex',gap:6}}>
              {[['⏮',()=>{setStep(0);setPlaying(false);}],
                ['◀',()=>setStep(s=>Math.max(0,s-1))],
                [playing?'⏸':'▶',()=>{if(step>=ops.length)setStep(0);setPlaying(p=>!p);}],
                ['▶',()=>setStep(s=>Math.min(ops.length,s+1))],
                ['⏭',()=>{setStep(ops.length);setPlaying(false);}],
              ].map(([label,fn],i)=>(
                <button key={i} onClick={fn} style={{
                  flex:1,background:label==='⏸'?'#2200aa':'#0a0a3a',
                  border:`1px solid ${label==='⏸'?'#6600ff':'#4400aa'}`,
                  color:label==='⏸'?'#cc44ff':'#00f0ff',
                  padding:'5px 0',borderRadius:4,cursor:'pointer',fontSize:'0.85em',fontFamily:'inherit',
                  textShadow:`0 0 8px ${label==='⏸'?'#cc44ff':'#00f0ff'}`,
                }}>{label}</button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

