import React, { useState } from 'react';
import { FileText, Folder, ChevronDown, ChevronRight, Code, Settings, Database, Brain, Activity, Eye, Cpu, Package } from 'lucide-react';

const ProjectStructure = () => {
  const [expanded, setExpanded] = useState({
    root: true,
    src: true,
    configs: true,
    data: true,
    models: false,
    logs: false,
    results: false
  });

  const toggle = (key) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const getIcon = (type) => {
    switch(type) {
      case 'folder': return <Folder className="w-4 h-4 text-blue-500" />;
      case 'python': return <Code className="w-4 h-4 text-green-500" />;
      case 'config': return <Settings className="w-4 h-4 text-orange-500" />;
      case 'doc': return <FileText className="w-4 h-4 text-gray-500" />;
      default: return <FileText className="w-4 h-4 text-gray-400" />;
    }
  };

  const files = {
    root: [
      { name: 'main.py', type: 'python', desc: 'Main pipeline orchestrator with CLI' },
      { name: 'requirements.txt', type: 'doc', desc: 'Python dependencies' },
      { name: 'README.md', type: 'doc', desc: 'Project documentation' },
      { name: 'Dockerfile', type: 'doc', desc: 'Docker containerization' },
      { name: '.gitignore', type: 'doc', desc: 'Git ignore rules' }
    ],
    src: [
      { name: 'data_loader.py', type: 'python', desc: 'DICOM loading & caching', icon: Database },
      { name: 'preprocessing.py', type: 'python', desc: 'Image preprocessing pipeline', icon: Activity },
      { name: 'feature_extraction.py', type: 'python', desc: 'Feature extraction methods', icon: Brain },
      { name: 'dataset.py', type: 'python', desc: 'PyTorch Dataset classes', icon: Database },
      { name: 'models.py', type: 'python', desc: '2D/3D model architectures', icon: Brain },
      { name: 'train.py', type: 'python', desc: 'Training loop & logic', icon: Cpu },
      { name: 'evaluate.py', type: 'python', desc: 'Evaluation & metrics', icon: Activity },
      { name: 'inference.py', type: 'python', desc: 'Model inference', icon: Eye },
      { name: 'utils.py', type: 'python', desc: 'Utility functions', icon: Package },
      { name: '__init__.py', type: 'python', desc: 'Package initialization', icon: Package }
    ],
    configs: [
      { name: 'config.yaml', type: 'config', desc: 'Main configuration file' },
      { name: 'model_configs.yaml', type: 'config', desc: 'Model-specific configs' },
      { name: 'preprocessing_configs.yaml', type: 'config', desc: 'Preprocessing presets' }
    ],
    data: [
      { name: 'raw/', type: 'folder', desc: 'Raw DICOM files' },
      { name: 'processed/', type: 'folder', desc: 'Preprocessed data cache' },
      { name: 'splits/', type: 'folder', desc: 'Train/val/test splits' }
    ]
  };

  const workflow = [
    { step: 1, name: 'Data Loading', module: 'data_loader.py', desc: 'Load DICOM files, extract metadata' },
    { step: 2, name: 'Preprocessing', module: 'preprocessing.py', desc: 'Normalize, denoise, augment' },
    { step: 3, name: 'Feature Extraction', module: 'feature_extraction.py', desc: 'Extract texture & deep features' },
    { step: 4, name: 'Dataset Preparation', module: 'dataset.py', desc: 'Create train/val/test splits' },
    { step: 5, name: 'Model Training', module: 'train.py', desc: 'Train with checkpointing & logging' },
    { step: 6, name: 'Evaluation', module: 'evaluate.py', desc: 'Compute metrics & visualize' },
    { step: 7, name: 'Inference', module: 'inference.py', desc: 'Deploy for predictions' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
            CBCT Dental Image Analysis Pipeline
          </h1>
          <p className="text-gray-300 text-lg">Production-Ready Deep Learning System for Medical Imaging</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Project Structure */}
          <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Folder className="w-6 h-6 text-blue-400" />
              Project Structure
            </h2>
            
            <div className="space-y-2 font-mono text-sm">
              <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-700/50 p-2 rounded" onClick={() => toggle('root')}>
                {expanded.root ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                <Folder className="w-4 h-4 text-blue-400" />
                <span className="font-semibold">cbct_pipeline/</span>
              </div>
              
              {expanded.root && (
                <div className="ml-6 space-y-2">
                  {files.root.map((file, i) => (
                    <div key={i} className="flex items-center gap-2 p-2 hover:bg-slate-700/30 rounded">
                      {getIcon(file.type)}
                      <span className="text-gray-300">{file.name}</span>
                      <span className="text-xs text-gray-500 ml-auto">{file.desc}</span>
                    </div>
                  ))}
                  
                  <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-700/50 p-2 rounded" onClick={() => toggle('src')}>
                    {expanded.src ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <Folder className="w-4 h-4 text-blue-400" />
                    <span className="font-semibold">src/</span>
                  </div>
                  
                  {expanded.src && (
                    <div className="ml-6 space-y-1">
                      {files.src.map((file, i) => (
                        <div key={i} className="flex items-center gap-2 p-1.5 hover:bg-slate-700/30 rounded text-xs">
                          {file.icon ? <file.icon className="w-3 h-3 text-green-400" /> : getIcon(file.type)}
                          <span className="text-gray-300">{file.name}</span>
                          <span className="text-gray-500 ml-auto">{file.desc}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-700/50 p-2 rounded" onClick={() => toggle('configs')}>
                    {expanded.configs ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    <Folder className="w-4 h-4 text-orange-400" />
                    <span className="font-semibold">configs/</span>
                  </div>
                  
                  {expanded.configs && (
                    <div className="ml-6 space-y-1">
                      {files.configs.map((file, i) => (
                        <div key={i} className="flex items-center gap-2 p-1.5 hover:bg-slate-700/30 rounded text-xs">
                          {getIcon(file.type)}
                          <span className="text-gray-300">{file.name}</span>
                          <span className="text-gray-500 ml-auto">{file.desc}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="flex items-center gap-2 p-2">
                    <Folder className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">data/</span>
                    <span className="text-xs text-gray-600 ml-auto">DICOM data directory</span>
                  </div>
                  <div className="flex items-center gap-2 p-2">
                    <Folder className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">saved_models/</span>
                    <span className="text-xs text-gray-600 ml-auto">Trained checkpoints</span>
                  </div>
                  <div className="flex items-center gap-2 p-2">
                    <Folder className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">logs/</span>
                    <span className="text-xs text-gray-600 ml-auto">Training logs</span>
                  </div>
                  <div className="flex items-center gap-2 p-2">
                    <Folder className="w-4 h-4 text-blue-400" />
                    <span className="text-gray-400">results/</span>
                    <span className="text-xs text-gray-600 ml-auto">Predictions & visualizations</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Pipeline Workflow */}
          <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Activity className="w-6 h-6 text-cyan-400" />
              Pipeline Workflow
            </h2>
            
            <div className="space-y-3">
              {workflow.map((item) => (
                <div key={item.step} className="flex gap-4 items-start p-3 bg-slate-700/30 rounded-lg hover:bg-slate-700/50 transition-colors">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center font-bold">
                    {item.step}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-white mb-1">{item.name}</h3>
                    <p className="text-xs text-gray-400 mb-1">{item.desc}</p>
                    <code className="text-xs text-green-400">{item.module}</code>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Key Features */}
        <div className="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <h2 className="text-2xl font-bold mb-4">Key Features</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-blue-400">Modular Design</h3>
              <p className="text-sm text-gray-300">Independent, testable components with clear interfaces</p>
            </div>
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-cyan-400">Production Ready</h3>
              <p className="text-sm text-gray-300">Complete error handling, logging, and validation</p>
            </div>
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-green-400">Easy Configuration</h3>
              <p className="text-sm text-gray-300">YAML-based config system, no hardcoded values</p>
            </div>
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-purple-400">Multiple Architectures</h3>
              <p className="text-sm text-gray-300">2D CNN, U-Net, 3D CNN for various tasks</p>
            </div>
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-yellow-400">GPU Optimized</h3>
              <p className="text-sm text-gray-300">Mixed precision, memory-efficient loading</p>
            </div>
            <div className="bg-slate-700/30 p-4 rounded-lg">
              <h3 className="font-semibold mb-2 text-red-400">Rich Visualization</h3>
              <p className="text-sm text-gray-300">TensorBoard integration, result overlays</p>
            </div>
          </div>
        </div>

        {/* Quick Start Commands */}
        <div className="mt-8 bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
          <h2 className="text-2xl font-bold mb-4">Quick Start Commands</h2>
          <div className="space-y-2 font-mono text-sm">
            <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
              <span className="text-gray-500"># Install dependencies</span><br/>
              <span className="text-green-400">pip install -r requirements.txt</span>
            </div>
            <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
              <span className="text-gray-500"># Train model</span><br/>
              <span className="text-green-400">python main.py --mode train --config configs/config.yaml</span>
            </div>
            <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
              <span className="text-gray-500"># Evaluate model</span><br/>
              <span className="text-green-400">python main.py --mode evaluate --checkpoint saved_models/best_model.pth</span>
            </div>
            <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
              <span className="text-gray-500"># Run inference</span><br/>
              <span className="text-green-400">python main.py --mode inference --input data/test/patient_001.dcm</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectStructure;