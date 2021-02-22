import datetime
import logging
import os
from pathlib import Path
import numpy as np

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
    self.parent.contributors = \
      ["Mahmoud Mostapha (UNC), Jared Vicory (Kitware), David Allemang (Kitware), Christian Herz (CHOP)"]
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
    uiWidget = slicer.util.loadUI(self.resourcePath(f'UI/{self.moduleName}.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # connect signals
    self.ui.InputDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.CommonSphereDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.FiducialsDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.OutputDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.OutputSphereDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.ProcrustesOutputDirectory.directoryChanged.connect(self.onUserInterfaceModified)
    self.ui.ProcrustesCheckBox.stateChanged.connect(lambda state: self.onUserInterfaceModified())
    self.ui.ApplyButton.clicked.connect(self.onApplyButton)

    # Refresh Apply button state
    logging.debug('0')
    self.onUserInterfaceModified()
    logging.debug('1')

    import os

    baseDir = '/Users/herzc/Documents/CHOP/SPHARM/DL_DATA_tricuspid/mid-systolic-data/GROUPS'
    leaflet = 'septal'

    self.ui.InputDirectory.directory = os.path.join(baseDir, leaflet, "in_models")
    self.ui.FiducialsDirectory.directory = os.path.join(baseDir, leaflet, "landmarks")
    self.ui.CommonSphereDirectory.directory = os.path.join(baseDir, leaflet, "common_unit_spheres")
    self.ui.OutputDirectory.directory = os.path.join(baseDir, leaflet, "out_models")
    self.ui.OutputSphereDirectory.directory = os.path.join(baseDir, leaflet, "out_spheres")
    self.ui.ProcrustesCheckBox.checked = True
    self.ui.ProcrustesOutputDirectory.directory = os.path.join(baseDir, leaflet, "proc_models")

  def onUserInterfaceModified(self):
    self.inputDir = Path(self.ui.InputDirectory.directory)
    self.commonSphereDir = Path(self.ui.CommonSphereDirectory.directory)
    self.fiducialsDir = Path(self.ui.FiducialsDirectory.directory)
    self.outputDir = Path(self.ui.OutputDirectory.directory)
    self.outputSphereDir = Path(self.ui.OutputSphereDirectory.directory)
    self.procrustesOutputDir = Path(self.ui.ProcrustesOutputDirectory.directory)
    self.ui.ProcrustesOutputDirectory.enabled = self.ui.ProcrustesCheckBox.checked

    # Check if each directory has been chosen
    # TODO: apply button needs to be enabled/disabled depending on all buttons
    self.ui.ApplyButton.enabled = '.' not in (self.inputDir, self.fiducialsDir, self.outputDir)

  def onApplyButton(self):
    models = self.inputDir.glob('*_pp_surf_SPHARM.vtk')
    fiducials = self.fiducialsDir.glob('*_fid.fcsv')
    unitSphere = next(self.commonSphereDir.glob('*_surf_para.vtk'))

    logic = RigidAlignmentModuleLogic()
    results = logic.run(
        models=models,
        fiducials=fiducials,
        unitSphere=unitSphere,
        outModelsDir=self.outputDir,
        outSphereDir=self.outputSphereDir,
        doProcrustes = self.ui.ProcrustesCheckBox.checked,
        procrustesDir = self.procrustesOutputDir
      )

    if results:
      self.showShapePopulationViewer(results)

  def showShapePopulationViewer(self, results):
    viewer = slicer.modules.shapepopulationviewer.widgetRepresentation()
    viewer.deleteModels()
    for polydata, name in results:
      viewer.loadModel(polydata, name)
    slicer.util.selectModule(slicer.modules.shapepopulationviewer)

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
    tempDir = Path(slicer.util.tempDirectory(key=self.moduleName))

    # TODO: it would be better to find files of corresponding cases
    models = sorted(models)
    fiducials = sorted(fiducials)

    bestTemplateIdx = self.getBestProcrustesTemplateIdx(models)

    aligned_models = sorted(self.runSphereProcrustes(models, fiducials, unitSphere, tempDir, tmplIndex=bestTemplateIdx))

    self.runRigidAlignment(tempDir, aligned_models, fiducials, unitSphere, outSphereDir)

    results = []
    for orig, fiducial, model in zip(models, fiducials, aligned_models):
      name = model.name.rsplit('_pp_surf_SPHARM', 1)[0]
      sphere = outSphereDir / f"{name}_rotSphere.vtk"
      outModel = outModelsDir / f"{name}_aligned.vtk"

      self.runSurfRemesh(sphere, model, unitSphere, outModel)
      res = self.buildColorMap(orig, fiducial, outModel)
      results.append(res)

    if results and doProcrustes:
      names = [Path(res[1]) for res in results]
      results = self.runProcrustes(names, procrustesDir, baseMeshIdx=bestTemplateIdx)

    return results

  def runSphereProcrustes(self, models, fiducials, unitSphereFileName, outputDir, tmplIndex=0):
    """ Run Procrustes alignment of the spheres
    """
    unitSpherePolyData = self.readPolydata(str(unitSphereFileName))
    unitSpherePoints = self.mapMeshPointsToSphere(models[tmplIndex], fiducials[tmplIndex], unitSpherePolyData)

    temp_models = []
    for modelFileName, fiducialFileName in zip(models, fiducials):
      modelMappedSpherePoints = self.mapMeshPointsToSphere(modelFileName, fiducialFileName, unitSpherePolyData)

      # rotate sphere
      R = self.getOptimalRotation(S=unitSpherePoints @ modelMappedSpherePoints.transpose())
      newSpherePolyData = self.getPolydataDeepCopy(unitSpherePolyData)
      for i in range(0, newSpherePolyData.GetNumberOfPoints()):
        pt = newSpherePolyData.GetPoint(i)
        npt = R @ pt
        newSpherePolyData.GetPoints().SetPoint(i,npt)

      # write rotated sphere
      namePrefix = modelFileName.name.split('_pp_surf_SPHARM')[0]
      outputRotSphereFileName = outputDir / f"{namePrefix}_rotSphere.vtk"
      outputModeFileName = outputDir / f"{namePrefix}_pp_surf_SPHARM.vtk"
      self.writePolydata(newSpherePolyData, outputRotSphereFileName)

      # Remesh original mesh using rotated sphere
      self.runSurfRemesh(outputRotSphereFileName, modelFileName, unitSphereFileName, outputModeFileName)

      temp_models.append(Path(outputModeFileName))

    logging.info(f'Tempmodels: {temp_models}')
    return temp_models

  def mapMeshPointsToSphere(self, meshFileName, fiducialsFileName, unitSpherePolyData):
    polydata = self.readPolydata(str(meshFileName))
    meshPts = self.getFiducialPoints(str(fiducialsFileName))
    mappedIndices = self.mapPointsToPolydataAndGetClosestIndices(polydata, meshPts)
    unitSpherePoints = self.getPolydataPointsByIndices(unitSpherePolyData, mappedIndices)
    return unitSpherePoints

  def getPolydataDeepCopy(self, polydata):
    newPolyData = vtk.vtkPolyData()
    newPolyData.DeepCopy(polydata)
    return newPolyData

  def getOptimalRotation(self, S):
    u, s, vh = np.linalg.svd(S)
    R = vh.transpose() @ u.transpose()
    if np.linalg.det(R) < 0:
      R[:, 2] = R[:, 2] * -1
    return R

  def getPolydataPointsByIndices(self, polydata, pointIndices):
    tspts = np.ndarray((3, len(pointIndices)))
    for i, mappedIdx in enumerate(pointIndices):
      pt = polydata.GetPoint(mappedIdx)
      tspts[0, i] = pt[0]
      tspts[1, i] = pt[1]
      tspts[2, i] = pt[2]
    return tspts

  def mapPointsToPolydataAndGetClosestIndices(self, polydata, pts):
    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    indices = [loc.FindClosestPoint(pt) for pt in pts]
    return indices

  def runProcrustes(self, models, outputDir, baseMeshIdx=0):
    """ run Procrustes by means of mesh points rather than fiducials only """
    logging.info('*****************************RUN PROCRUSTES*****************************')

    baseMesh = self.readPolydata(str(models[baseMeshIdx]))
    nPoints = baseMesh.GetNumberOfPoints()

    Y, Y_mean, Ymm = self.getMeshAttributes(baseMesh, nPoints)

    results = []
    for model in models:
      logging.info(f"Processing {model}")
      mesh = self.readPolydata(str(model))

      X, X_mean, Xmm = self.getMeshAttributes(mesh, nPoints)
      R = self.getOptimalRotation(S=Xmm.transpose() @ Ymm)

      err_after, err_before = self.calcErrorBeforeAndAfter(nPoints, R, X_mean, Y, Y_mean, X)
      logging.info(f"Error reduced from {err_before} to {err_after}")

      new_mesh = self.getPolydataDeepCopy(mesh)
      for i in range(0, nPoints):
          point = new_mesh.GetPoint(i)
          new_point = R @ (point - X_mean) + Y_mean
          new_mesh.GetPoints().SetPoint(i,new_point)

      name, ext = os.path.splitext(model.name)
      outname = os.path.join(str(outputDir), name + '_proc2.vtk')
      logging.info(f"   Output: {outname}")

      results.append((new_mesh, outname))
      self.writePolydata(new_mesh, outname)

    return results

  def getBestProcrustesTemplateIdx(self, models):
    """ run Procrustes by means of mesh points rather than fiducials only """

    best = np.inf
    best_idx = np.inf

    for baseIdx, base in enumerate(models):
      baseMesh = self.readPolydata(str(base))
      nPoints = baseMesh.GetNumberOfPoints()
      Y, Y_mean, Ymm = self.getMeshAttributes(baseMesh, nPoints)

      avg_err_after = []
      for model in models:
        mesh = self.readPolydata(str(model))

        X, X_mean, Xmm = self.getMeshAttributes(mesh, nPoints)
        R = self.getOptimalRotation(S=Xmm.transpose() @ Ymm)

        err_after, err_before = self.calcErrorBeforeAndAfter(nPoints, R, X_mean, Y, Y_mean, X)
        avg_err_after.append(err_after)

      if np.array(avg_err_after).mean() < best:
        best = np.array(avg_err_after).mean()
        best_idx = baseIdx
        logging.info(f"New best model template index: {best_idx}, {models[best_idx]}")

    logging.info(f"Best model template index: {best_idx}, {models[best_idx]}")
    return best_idx

  def calcErrorBeforeAndAfter(self, nPoints, R, X_mean, Y, Y_mean, X):
    Xa = np.ndarray(shape=(nPoints, 3), dtype=float)
    for i in range(0, nPoints):
      Xa[i, :] = R @ (X[i, :] - X_mean) + Y_mean
    err_before = np.sum(np.sum((X - Y) ** 2, axis=0))
    err_after = np.sum(np.sum((Xa - Y) ** 2, axis=0))
    return err_after, err_before

  def getMeshAttributes(self, mesh, nPoints):
    Y = np.ndarray(shape=(nPoints, 3), dtype=float)
    for i in range(nPoints):
      pt = mesh.GetPoint(i)
      Y[i, 0] = pt[0]
      Y[i, 1] = pt[1]
      Y[i, 2] = pt[2]
    Y_mean = np.mean(Y, axis=0)
    Ymm = Y - Y_mean
    return Y, Y_mean, Ymm

  def getFiducialPoints(self, fileName):
    pts = []
    with open(fileName) as tfid:
      lines = tfid.readlines()
      for line in lines:
        if line[0] != '#':
          s = line.split(',')
          pt = [float(s[1]), float(s[2]), float(s[3])]
          pts.append(pt)
    return np.array(pts)

  @staticmethod
  def readPolydata(fileName):
    polyReader = vtk.vtkPolyDataReader()
    polyReader.SetFileName(fileName)
    polyReader.Update()
    return polyReader.GetOutput()

  @staticmethod
  def writePolydata(polydata, fileName):
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(str(fileName))
    w.SetInputData(polydata)
    w.Write()

  def runRigidAlignment(self, tempDir, models, fiducials, sphere, outputDir):
    inputCSV = tempDir / f'{datetime.datetime.now().isoformat()}.csv'
    with inputCSV.open('w', newline='') as f:
      for i in range(len(models)):
        row = str(models[i]) + ',' + str(fiducials[i])
        line = ''.join(row) + '\n'
        f.write(line)

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

  def buildColorMap(self, origModel, fiducial, outModel):
    orig_mesh = self.readPolydata(str(origModel))

    origPhiArray = orig_mesh.GetPointData().GetScalars("_paraPhi")

    new_mesh = self.readPolydata(str(outModel))
    new_mesh.GetPointData().SetActiveScalars("_paraPhi")
    new_mesh.GetPointData().SetScalars(origPhiArray)
    new_mesh.Modified()

    ptArray = vtk.vtkDoubleArray()
    ptArray.SetNumberOfComponents(1)
    ptArray.SetNumberOfValues(new_mesh.GetNumberOfPoints())
    ptArray.SetName('Landmarks')
    for ind in range(0, ptArray.GetNumberOfValues()):
      ptArray.SetValue(ind, 0.0)

    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(new_mesh)
    loc.BuildLocator()

    pts = self.getFiducialPoints(fiducial)
    for l_ind in range(0, len(pts)):
      ind = loc.FindClosestPoint(pts[l_ind])
      ptArray.SetValue(ind, l_ind + 1)

    new_mesh.GetPointData().AddArray(ptArray)

    self.writePolydata(new_mesh, str(outModel))

    return new_mesh, str(outModel)