import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Initialization data for the jupyterlab-autoplay extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyterlab-autoplay:plugin',
  description: 'Automatically run and/or hide cells when opening a Jupyter notebook',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension jupyterlab-autoplay is activated!');
  }
};

export default plugin;
