define([
    'base/js/namespace',
    'base/js/events'
], function(
    Jupyter,
    events
) {
    // Ajoutez une fonction pour cacher le code
    var hide_code = function() {
        Jupyter.notebook.get_cells().forEach(function(cell) {
            if (cell.cell_type === 'code') {
                cell.element.find('div.input').hide();
            }
        });
    };

    // Exécutez la fonction lorsque le notebook est entièrement chargé
    function load_ipython_extension() {
        if (Jupyter.notebook !== undefined && Jupyter.notebook._fully_loaded) {
            hide_code();
        }
        events.on("notebook_loaded.Notebook", hide_code);
    }

    return {
        load_ipython_extension: load_ipython_extension
    };
});