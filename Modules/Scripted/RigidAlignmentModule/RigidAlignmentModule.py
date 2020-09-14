import csv
import datetime
import glob
import logging
import os
import pathlib
import numpy as np
import time

import qt
import slicer
import slicer.util
import slicer.cli
from slicer.ScriptedLoadableModule import *
import vtk


#
# RigidAlignmentModule
#

class RigidAlignmentModule(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SPHARM-PDM Correspondence Improvement"
    self.parent.categories = ["Shape Creation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Mahmoud Mostapha (UNC), Jared Vicory (Kitware), David Allemang (Kitware)"]
    self.parent.helpText = """
    Rigid alignment of the landmarks on the unit sphere: the input models share the same unit sphere 
    and their landmarks are defined as spacial coordinates (x,y,z) of the input model. 
    """
    self.parent.acknowledgementText = """
      This work was supported by NIH NIBIB R01EB021391
      (Shape Analysis Toolbox for Medical Image Computing Projects).
    """


#
# RigidAlignmentModuleWidget
#

class RigidAlignmentModuleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    #
    #  Interface
    #
    loader = qt.QUiLoader()
    self.moduleName = 'RigidAlignmentModule'
    scriptedModulesPath = eval('slicer.modules.%s.path' % self.moduleName.lower())
    scriptedModulesPath = os.path.dirname(scriptedModulesPath)
    path = os.path.join(scriptedModulesPath, 'Resources', 'UI', '%s.ui' % self.moduleName)
    qfile = qt.QFile(path)
    qfile.open(qt.QFile.ReadOnly)
    widget = loader.load(qfile, self.parent)
    self.layout = self.parent.layout()
    self.widget = widget
    self.layout.addWidget(widget)
    self.ui = slicer.util.childWidgetVariables(widget)

    # Connections
    # Directories
    self.ui.InputDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    self.ui.CommonSphereDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    self.ui.FiducialsDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    self.ui.OutputDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    self.ui.OutputSphereDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    self.ui.ProcrustesOutputDirectory.connect('directoryChanged(const QString &)', self.onSelect)
    # Procrustes
    self.ui.ProcrustesCheckBox.connect('stateChanged(int)',self.onProcrustesChecked)
    #   Apply CLIs
    self.ui.ApplyButton.connect('clicked(bool)', self.onApplyButton)

    # Refresh Apply button state
    print('0')
    self.onSelect()
    print('1')
    self.onProcrustesChecked()
    print('2')

  def cleanup(self):
    pass

  #
  #   Directories
  #
  def onSelect(self):
    self.inputDir = pathlib.Path(self.ui.InputDirectory.directory)
    self.commonSphereDir = pathlib.Path(self.ui.CommonSphereDirectory.directory)
    self.fiducialsDir = pathlib.Path(self.ui.FiducialsDirectory.directory)
    self.outputDir = pathlib.Path(self.ui.OutputDirectory.directory)
    self.outputSphereDir = pathlib.Path(self.ui.OutputSphereDirectory.directory)
    self.procrustesOutputDir = pathlib.Path(self.ui.ProcrustesOutputDirectory.directory)

    # Check if each directory has been choosen
    self.ui.ApplyButton.enabled = '.' not in (self.inputDir, self.fiducialsDir, self.outputDir)

  def onProcrustesChecked(self):
    self.ui.ProcrustesOutputDirectory.enabled = self.ui.ProcrustesCheckBox.checked

  def onApplyButton(self):
    models = self.inputDir.glob('*_pp_surf_SPHARM.vtk')
    fiducials = self.fiducialsDir.glob('*_fid.fcsv')
    unitSphere = next(self.commonSphereDir.glob('*_surf_para.vtk'))

    logic = RigidAlignmentModuleLogic()
    logic.run(
      models=models,
      fiducials=fiducials,
      unitSphere=unitSphere,
      outModelsDir=self.outputDir,
      outSphereDir=self.outputSphereDir,
      doProcrustes = self.ui.ProcrustesCheckBox.checked,
      procrustesDir = self.procrustesOutputDir
    )

#
# RigidAlignmentModuleLogic
#

class RigidAlignmentModuleLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def run(self, models, fiducials, unitSphere, outModelsDir, outSphereDir, doProcrustes, procrustesDir):
    """
    Note that all paths are expected to be pathlib paths.

    models: A sequence of paths to SPHARM model files. (*_pp_surf_SPHARM.vtk)
    fiducials: A sequence of paths to fiducial data files. (*_fid.fcsv)
    unitSphere: A path to a unit sphere for alignment. (*_surf_para.vtk)
    outputDir: Output directory for aligned spheres.
    """

    models = sorted(models)
    fiducials = sorted(fiducials)

    temp = pathlib.Path(slicer.util.tempDirectory(key='RigidAlignment'))

    # Use procrustes pre-processing to align spheres based on fiducials
    aligned_models = sorted(self.sphere_procrustes(models,fiducials,unitSphere,temp))

    # Use aligned models as inputs to RA
    now = datetime.datetime.now().isoformat()
    inputCSV = temp / '{}.csv'.format(now)

    with inputCSV.open('w', newline='') as f:
      for i in range(len(aligned_models)):
        row = str(aligned_models[i]) + ',' + str(fiducials[i])
        line = ''.join(row) + '\n'
        f.write(line)

    self.runRigidAlignment(inputCSV, unitSphere, outSphereDir)

    results = []

    for i in range(len(models)):
      model = aligned_models[i]
      fiducial = fiducials[i]
      orig = models[i]

      name = model.name.rsplit('_pp_surf_SPHARM', 1)[0]
      sphere = os.path.join(outSphereDir, name + '_rotSphere.vtk')
      outModel = os.path.join(outModelsDir, name + '_aligned.vtk')

      self.runSurfRemesh(sphere, model, unitSphere, outModel)
      res = self.buildColorMap(model, orig, fiducial, outModel)

      results.append(res)

    if doProcrustes:
      names = [pathlib.Path(res[1]) for res in results]

      print(procrustesDir)
      results = self.procrustes(names,procrustesDir)

    if results:
      self.showViewer(results)

  def procrustes(self,models,outputDir):
    print('*******************************************PROCRUSTES')
    print(outputDir)
    results = []

    tr = vtk.vtkPolyDataReader()
    tr.SetFileName(str(models[0]))
    tr.Update()
    base_mesh = tr.GetOutput()

    name,ext = os.path.splitext(models[0].name)
    outname = os.path.join(str(outputDir), name + '_proc2.vtk')

    npoints = base_mesh.GetNumberOfPoints()

    Y = np.ndarray(shape=(npoints,3), dtype=float)
    for i in range(npoints):
      pt = base_mesh.GetPoint(i)
      Y[i,0] = pt[0]
      Y[i,1] = pt[1]
      Y[i,2] = pt[2]

    Y_mean = np.mean(Y,axis=0)
    Ymm = Y - Y_mean

    results.append((base_mesh,outname))

    for model in models[1:]:
      print(model)
      r = vtk.vtkPolyDataReader()
      r.SetFileName(str(model))
      r.Update()
      mesh = r.GetOutput()
      
      X = np.ndarray(shape=(npoints,3), dtype=float)
      for i in range(npoints):
        pt = mesh.GetPoint(i)
        X[i,0] = pt[0]
        X[i,1] = pt[1]
        X[i,2] = pt[2]
              
      X_mean = np.mean(X,axis=0)
      Xmm = X - X_mean
          
      S = Xmm.transpose() @ Ymm
      u,s,vh = np.linalg.svd(S)

      R = vh.transpose() @ u.transpose()
      if (np.linalg.det(R) < 0):
          R[:,2] = R[:,2]*-1
      t = Y_mean - R @ X_mean
      
      new_mesh = vtk.vtkPolyData()
      new_mesh.DeepCopy(mesh)

      for i in range(0,npoints):
          point = new_mesh.GetPoint(i)
          new_point = R @ (point - X_mean) + Y_mean
          new_mesh.GetPoints().SetPoint(i,new_point)
      
      name,ext = os.path.splitext(model.name)
      outname = os.path.join(str(outputDir), name + '_proc2.vtk')
      print(outname)

      results.append((new_mesh, outname))
      
      w = vtk.vtkPolyDataWriter()
      w.SetInputData(new_mesh)
      w.SetFileName(outname)
      w.Write()

      Xa = np.ndarray(shape=(npoints,3), dtype=float)
      for i in range(0,npoints):
          Xa[i,:] = R @ (X[i,:] - X_mean) + Y_mean
      
      err_before = np.sum( np.sum((X-Y)**2,axis=0) )
      err_after = np.sum( np.sum((Xa-Y)**2,axis=0) )
      
      print("Error reduced from %f to %f" % (err_before, err_after))

    return results
    
  def sphere_procrustes(self, models, fiducials, unitSphere, outputDir):
    # Do procrustes alignment of the spheres 

    # Use models[0] as the template
    tr = vtk.vtkPolyDataReader()
    tr.SetFileName(str(models[0]))
    tr.Update()
    template = tr.GetOutput()

    sr = vtk.vtkPolyDataReader()
    sr.SetFileName(str(unitSphere))
    sr.Update()
    sphere = sr.GetOutput()

    # Map fiducials to sphere
    tfid = open(str(fiducials[0]))
    lines = tfid.readlines()
    pts = []
    for line in lines:
      if line[0] != '#':
        s = line.split(',')
        pt = [ float(s[1]), float(s[2]), float(s[3])]
        pts.append(pt)

    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(template)
    loc.BuildLocator()

    tpts = np.array(pts)
    tinds = []
    for i in range(len(pts)):
      ind = loc.FindClosestPoint(pts[i])
      tinds.append(ind)

    tspts = np.ndarray((3,len(pts)))
    for i in range(len(tinds)):
      pt = sphere.GetPoint(tinds[i])
      tspts[0,i] = pt[0]
      tspts[1,i] = pt[1]
      tspts[2,i] = pt[2]

    # Loop over models
    temp_models = []
    for mi in range(0,len(models)):
      r = vtk.vtkPolyDataReader()
      r.SetFileName(str(models[mi]))
      r.Update()
      m = r.GetOutput()

      fid = open(str(fiducials[mi]))
      lines = fid.readlines()
      pts = []
      for line in lines:
        if line[0] != '#':
          s = line.split(',')
          pt = [ float(s[1]), float(s[2]), float(s[3])]
          pts.append(pt)

      loc = vtk.vtkKdTreePointLocator()
      loc.SetDataSet(m)
      loc.BuildLocator()

      mpts = np.array(pts)
      inds = []
      for i in range(len(pts)):
        ind = loc.FindClosestPoint(pts[i])
        inds.append(ind)

      # get points from sphere
      spts = np.ndarray((3,len(pts)))
      for i in range(len(inds)):
        pt = sphere.GetPoint(inds[i])
        spts[0,i] = pt[0]
        spts[1,i] = pt[1]
        spts[2,i] = pt[2]

      # compute optimal rotation
      S = tspts @ spts.transpose()
      u,s,vh = np.linalg.svd(S)
      R = vh.transpose() @ u.transpose()

      # rotate sphere
      nsphere = vtk.vtkPolyData()
      nsphere.DeepCopy(sphere)

      for i in range(0,nsphere.GetNumberOfPoints()):
        pt = nsphere.GetPoint(i)
        npt = R @ pt
        nsphere.GetPoints().SetPoint(i,npt)

      # write rotated sphere
      name = models[mi].name.split('_pp_surf_SPHARM')[0]
      outname = os.path.join(str(outputDir),name + '_rotSphere.vtk')

      w = vtk.vtkPolyDataWriter()
      w.SetFileName(outname)
      w.SetInputData(nsphere)
      w.Update()

      outModel = os.path.join(str(outputDir),name+'_pp_surf_SPHARM.vtk')
      temp_models.append(pathlib.Path(outModel))

      # Remesh original mesh using rotated sphere
      self.runSurfRemesh(outname, models[mi], unitSphere, outModel)

    print(f'Tempmodels: {temp_models}')
    return temp_models

  def runRigidAlignment(self, inputCSV, sphere, outputDir):
    args = {
      'inputCSV': str(inputCSV),
      'sphere': str(sphere),
      'output': str(outputDir)
    }

    logging.info('Launching RigidAlignment Module.')
    slicer.cli.run(slicer.modules.rigidalignment, None, args, wait_for_completion=True)
    logging.info('RigidAlignment Completed.')

  def runSurfRemesh(self, sphere, model, unitSphere, outModel):
    args = {
      'temp': str(sphere),
      'input': str(model),
      'ref': str(unitSphere),
      'output': str(outModel),
      'keepColor': True
    }

    logging.info('Launching SRemesh Module.')
    slicer.cli.run(slicer.modules.sremesh, None, args, wait_for_completion=True)
    logging.info('SRemesh Completed.')

  def buildColorMap(self, model, origModel, fiducial, outModel):
    reader_in = vtk.vtkPolyDataReader()
    reader_in.SetFileName(str(model))
    reader_in.Update()
    init_mesh = reader_in.GetOutput()

    reader_orig = vtk.vtkPolyDataReader()
    reader_orig.SetFileName(str(origModel))
    reader_orig.Update()
    orig_mesh = reader_orig.GetOutput()
    phiArray = orig_mesh.GetPointData().GetScalars("_paraPhi")

    reader_out = vtk.vtkPolyDataReader()
    reader_out.SetFileName(str(outModel))
    reader_out.Update()
    new_mesh = reader_out.GetOutput()
    new_mesh.GetPointData().SetActiveScalars("_paraPhi")
    new_mesh.GetPointData().SetScalars(phiArray)
    new_mesh.Modified()

    with open(fiducial) as fid:
      lines = fid.readlines()
      pts = []
      for line in lines:
        if line[0] != '#':
          s = line.split(',')
          pt = [float(s[1]), float(s[2]), float(s[3])]
          pts.append(pt)

    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(new_mesh)
    loc.BuildLocator()

    ptArray = vtk.vtkDoubleArray()
    ptArray.SetNumberOfComponents(1)
    ptArray.SetNumberOfValues(new_mesh.GetNumberOfPoints())
    ptArray.SetName('Landmarks')
    for ind in range(0, ptArray.GetNumberOfValues()):
      ptArray.SetValue(ind, 0.0)

    for l_ind in range(0, len(pts)):
      ind = loc.FindClosestPoint(pts[l_ind])
      ptArray.SetValue(ind, l_ind + 1)

    new_mesh.GetPointData().AddArray(ptArray)

    # write results
    polyDataWriter = vtk.vtkPolyDataWriter()
    polyDataWriter.SetInputData(new_mesh)
    polyDataWriter.SetFileName(str(outModel))
    polyDataWriter.Write()

    return new_mesh, str(outModel)

  def showViewer(self, results):
    viewer = slicer.modules.shapepopulationviewer.widgetRepresentation()
    viewer.deleteModels()
    for polydata, name in results:
      viewer.loadModel(polydata, name)
    slicer.util.selectModule(slicer.modules.shapepopulationviewer)
