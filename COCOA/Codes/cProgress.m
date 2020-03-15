% CPROGRESS - a circular progress bar (with 2 circular progress notifications)
%
%    Similar to waitbar but has 2 circular 
%
%   Initialise
%   ----------
%     h = CPROGRESS ( initValue, userText, optArg, argValue );
%
%      initValue  - the initial value (normally 0);
%                 - or 'busy' -> this indicates a unknown process length
%      updateText - update the user controlled text string.
%      optArg     - see valid options below
%      argValue   - the value
%
%      % Valid optArgs to cusomise the GUI:
%         options.innerBar   = false;         % inc inner bar true | false
%         options.outerColor = [0 1 0];       % outer bar colour
%         options.innerColor = [0 0.8 0];     % inner bar colour
%         options.edgeColor  = [0.7 0.7 0.7]; % edge colour
%         options.fade       = true;          % fade older bars true | false
%         options.forceFocus = false;         % force focus of ui on each call
%         options.parent     = [];            % embed in a user GUI component
%     Normal mode only
%         options.estTime    = false;         % include an time estimate
%     Busy mode only:
%         options.timer      = true;          % use with busy mode only
%         options.dTime      = 0.1;           % time between timer calls
%
%        To extract the most upto date list of options:
%         options = CPROGRESS;
%
%
%   Update at runtime:
%   ------------------
%      CPROGRESS ( value, h );
%      CPROGRESS ( value, h, updateText );
%
%      value      - value in percent that bar should be displayed
%      updateText - update the user controlled text string.
%
%    When using a double progress bar:
%
%      CPROGRESS ( [outerValue innerValue], h );
%
%      outerValue - value in percent that outer bar should be displayed
%      innerValue - value in percent that inner bar should be displayed
%                 - only valid if bar initialised at start by using
%                   the "innerBar" optArg pair (see example)
%
%   Run internal demo:
%   ------------------
%    cProgress ( 'demo' );
%    cProgress ( 'embedDemo' );
%    cProgress ( 'busyDemo' );
%    cProgress ( 'busyTimer' );
%
%   Stand alone example:
%   --------
%   Create the dialog specifying that the inner bar is included:
%   h = cProgress (0, 'Running Demo...', 'innerBar', true );
%    
%   Update the bars [outer inner]
%   cProgress ( [25 40], h )
%
%   Busy Example:
%   -------------
%   h = cProgress ('busy', 'Close to quit' );
%   while ( true )
%     % Your code goes here
%     cProgress ( 'busy', h );
%     % Some condition goes here
%     %  to break the while
%     %
%     if ~ishandle ( h ); break ;end
%     pause ( 0.025 );
%   end
%
%   Embed Example:
%   --------------
%    f = figure; 
%    uip = uipanel ( 'parent', f, 'position', [0.2 0.2 0.4 0.4] );
%    h = cProgress ( 0, 'Embedded', 'parent', uip, 'outerColor', [0 0 1] )
%    for i=1:100
%      % Your code goes here...
%      cProgress ( i, h )
%    end
%    % Clean up the uipanel when its finished
%    delete ( h );
%
%   Notes:
%   ------
%
%   1. If the dialog is closed before your calling loop is finished it 
%      will continue to run (but not display)
%   2. Dont make too many calls to this function -> it can slow down
%      the over all progess of your code.
%   3. You can embed the progress bar into your own GUI by creating
%      a new panel for the bar to be displayed in.
%   4. For a busy progress bar -> the user must indicate when it is to 
%      be updated - you can initiate this via a timer if you desire
%
%
%  see also waitbar % Not my program at all.
%
% $Id: cProgress.m 250 2015-10-05 19:57:02Z Bob $
function varargout = cProgress ( varargin )
  try
    if nargin == 1 && ischar ( varargin{1} ) && ...
        ( strcmp ( varargin{1}, 'demo' ) || strcmp ( varargin{1}, 'embedDemo' ) || strcmp ( varargin{1}, 'busyDemo' ) || strcmp ( varargin{1}, 'busyTimer' ) )
      cProgressDemo ( varargin{1} );
      return
    end
    % set up the defaults
    options.innerBar = false;
    options.outerColor = [0 1 0];
    options.innerColor = [0 0.8 0];
    options.edgeColor  = [0.7 0.7 0.7];
    options.fade       = true;
    options.forceFocus = false;
    options.estTime    = false;
    options.parent     = [];
    options.timer      = false;
    options.dTime      = 0.1;
    if nargin == 0 && nargout == 1;
      varargout{1} = options;
      return
    end
    
    % determine whether this is an update call.,
    init = true;
    if length ( varargin ) >= 2 && ~ischar ( varargin{2} )
      init = false;
    end
    if init % create the figure
      if isnumeric ( varargin{1} ) && nargout ~= 1
        error ( 'cProgress:nargout', 'You must save the returned handle from the function' );
      end
      for ii=3:2:length(varargin)
        options.(varargin{ii}) = varargin{ii+1};
      end
      if isempty ( options.parent )
        f = handle ( figure ( 'ToolBar','none', 'MenuBar','none', 'Name', 'Please Wait', 'NumberTitle', 'off', 'resize', 'off') );
        f.Position(4) = 175;
        f.Position(3) = f.Position(4)-20;
        centerfig(f);
      else
        f = options.parent;
      end
      f = handle(f);
      ax = handle ( axes ( 'parent', f, 'Position', [0 0 1 1], 'Visible', 'off', 'XTick', [], 'YTick', [], 'nextplot', 'add' ) );
      ax.XLimMode = 'manual';
      ax.YLimMode = 'manual';
      ax.Units = 'pixels';
      dPos = ax.Position(4)-ax.Position(3);
      % Make sure the axes is square...
      if ax.Position(3) ~= ax.Position(4)
        ax.Position(3) = min(ax.Position(3:4));
        ax.Position(4) = ax.Position(3);
        if dPos < 0
          ax.Position(1) = ax.Position(1) - dPos/2;
        else
          ax.Position(2) = ax.Position(2) + dPos/2;
        end
      end
      ax.YLim = [-0.1 1];
      drawArc ( ax, [0.5 0.5], 0.45, -pi:0.01:pi, options.edgeColor );
      drawArc ( ax, [0.5 0.5], 0.33, -pi:0.01:pi, options.edgeColor );
      if options.innerBar
        drawArc ( ax, [0.5 0.5], 0.25, -pi:0.01:pi, options.edgeColor );
      end
      if nargin >=2; userTxt = varargin{2}; else userTxt = 'Working...'; end
      t = handle ( text ( 0.5, 0.5, userTxt, 'HorizontalAlignment', 'center' ) );
      varargout{1} = f;
      f.ApplicationData.matpi.cProgress.userText = t;
      f.ApplicationData.matpi.cProgress.inner.flag = options.innerBar;
      f.ApplicationData.matpi.cProgress.outer.lastPos = 0;
      f.ApplicationData.matpi.cProgress.inner.lastPos = 0;
      f.ApplicationData.matpi.cProgress.ax = ax;
      f.ApplicationData.matpi.cProgress.outer.h = [];
      f.ApplicationData.matpi.cProgress.inner.h = [];
      f.ApplicationData.matpi.cProgress.outer.stepSize = 0;
      f.ApplicationData.matpi.cProgress.options = options;
      f.ApplicationData.matpi.cProgress.busy = false;
      if options.estTime
        f.ApplicationData.matpi.cProgress.start = now;
        f.ApplicationData.matpi.cProgress.estTime = handle ( text ( 0.5, -0.03, '', 'HorizontalAlignment', 'center' ) );
      end
      % if not initialised at 0 -> self call to initialise the start position 
      if varargin{1} ~= 0
        cProgress ( varargin{1}, f )
      else
        drawnow();
      end
      if options.timer && ischar ( varargin{1} ) && strcmp ( varargin{1}, 'busy' )
        f.ApplicationData.matpi.cProgress.options.estTime = false;
        t = timer('TimerFcn', @(a,b)cProgress ( 'busy', f ), 'Period', options.dTime, 'ExecutionMode', 'FixedRate' );
        start ( t );
        addlistener ( f, 'ObjectBeingDestroyed', @(a,b)killTimer(t) );
      end
    else
      f = handle ( varargin{2} );
      outer = varargin{1};
      if nargin == 3 && ischar ( varargin{3} )
        f.ApplicationData.matpi.cProgress.userText.String = varargin{3};
      end
      try innerflag = f.ApplicationData.matpi.cProgress.inner.flag; catch le; innerflag = false; end
      if ischar ( outer ) && strcmp ( outer, 'busy' )
        outer = f.ApplicationData.matpi.cProgress.outer.lastPos + 2;
        if outer > 100
          outer = 2;
          f.ApplicationData.matpi.cProgress.busy = true;
          f.ApplicationData.matpi.cProgress.options.estTime = false;
        end
      end
      if length ( outer ) > 1
        inner = max(0,min(outer(2),100));
        outer = max(0,min(outer(1),100));
      else
        inner = 0;
        innerflag = false;
      end
      rads = [0.33 0.45];
      updateGui ( f, outer, 'outer', rads )
      if innerflag
        rads = [0.25 0.33];
        updateGui ( f, inner, 'inner', rads )
      end
      % Estimate the anount of time lest to progress the data
      %   calc based on elapsed time since start and the current
      %   position of the bar(s).
      if f.ApplicationData.matpi.cProgress.options.estTime
        if f.ApplicationData.matpi.cProgress.outer.stepSize == 0
          cPos = outer + inner/100;
        else
          cPos = outer + (f.ApplicationData.matpi.cProgress.outer.stepSize*inner)/100;
        end
        seconds    = (now-f.ApplicationData.matpi.cProgress.start)*(24*3600);
        totalTime  = 100*(seconds/cPos);
        finishTime = max(0,totalTime-seconds);
        if finishTime < 60
          f.ApplicationData.matpi.cProgress.estTime.String = sprintf ( 'Est finish in %.0f second(s)', finishTime );
        elseif finishTime < 3600
          f.ApplicationData.matpi.cProgress.estTime.String = sprintf ( 'Est finish in %.0f minute(s)', finishTime/60 );
        else
          f.ApplicationData.matpi.cProgress.estTime.String = sprintf ( 'Est finish in %.0f hour(s)', finishTime/3600 );
        end
      end
      if f.ApplicationData.matpi.cProgress.options.forceFocus
        figure ( ancestor ( f, 'figure' ) );
      end
    end
  catch le
    % Catch construction errors
    switch le.identifier
      case 'cProgress:nargout'
        rethrow(le)
    end      
  end
end  
function updateGui ( f, percent, mode, rads )
  ax = f.ApplicationData.matpi.cProgress.ax;
  lastPos = f.ApplicationData.matpi.cProgress.(mode).lastPos;
  percent = ceil(percent);
  if percent == lastPos; return; end
  switch mode
    case 'outer'
      color = f.ApplicationData.matpi.cProgress.options.outerColor;
      f.ApplicationData.matpi.cProgress.outer.stepSize = percent-lastPos;
    otherwise
      if percent < lastPos % reset if percent < lastPos
        lastPos = 0;
        delete ( f.ApplicationData.matpi.cProgress.inner.h );
        f.ApplicationData.matpi.cProgress.inner.lastPos = 0;
        f.ApplicationData.matpi.cProgress.inner.h = [];
      end
      color = f.ApplicationData.matpi.cProgress.options.innerColor;
  end
  steps = fliplr(linspace ( -1.5*pi, 0.5*pi, 100 ));
  % Check the last pos is an integer
  if f.ApplicationData.matpi.cProgress.busy
    f.ApplicationData.matpi.cProgress.(mode).h = [f.ApplicationData.matpi.cProgress.(mode).h(2:end) f.ApplicationData.matpi.cProgress.(mode).h(1)];
  else
    lastPos = floor(max(1,lastPos));
    [iX, iY] = calcArc ( [0.5 0.5], rads(1), steps(lastPos:percent) );
    [oX, oY] = calcArc ( [0.5 0.5], rads(2), steps(lastPos:percent) );
    xx = [iX(1) iX(2:end) fliplr(oX) iX(1)];
    yy = [iY(1) iY(2:end) fliplr(oY) iY(1)];
    h = patch ( xx, yy, color, 'EdgeColor', color, 'Parent', ax );
    f.ApplicationData.matpi.cProgress.(mode).h(end+1) = h;
  end
  f.ApplicationData.matpi.cProgress.(mode).lastPos = percent;
  tValue = 1;
  if percent-lastPos < 10
    dimStep = 0.1;
  else
    dimStep = 0.2;
  end
  if f.ApplicationData.matpi.cProgress.options.fade
    for ii=length(f.ApplicationData.matpi.cProgress.(mode).h)-1:-1:1
      tValue = tValue-dimStep;
      if tValue < 0.2
        break
      end
      h = handle ( f.ApplicationData.matpi.cProgress.(mode).h(ii) );
      h.FaceAlpha = tValue;
      h.EdgeAlpha = tValue;
    end
  end
  drawnow();
end
function [x, y] = calcArc ( xy, radius, arc )
  x=xy(1)+radius*cos(arc);
  y=xy(2)+radius*sin(arc);
end
function h = drawArc ( ax, xy, radius, arc, color )
  [x, y] = calcArc ( xy, radius, arc );
  h = plot ( ax, x, y, 'Color', color );
end
function killTimer ( t )
  stop(t);
  delete(t);
end
% Some in built demos...
function cProgressDemo ( mode )
  switch mode
    case 'demo'
      disp ( 'create a demo with 2 bars' );
      h = cProgress (0, sprintf ( 'cProgress Demo\nwww.matpi.com' ), 'innerBar', true );
    case 'embedDemo'
      disp ( 'create a demo with 2 bars embeded in another GUI' );
      f = figure;
      uip = uipanel ( 'parent', f, 'position', [0.7 0.7 0.3 0.3] );
      h = cProgress (0, 'Running Demo...', 'innerBar', true, 'parent', uip );      
    case 'busyDemo'
      busyExample();
      return
    case 'busyTimer'
      busyTimerExample();
      return
    otherwise
      return
  end
  for i=1:25
    for j=1:50
      cProgress ( 100*[(i-1)/25 j/50], h )
    end
    cProgress ( 100*[i/25 0], h, sprintf ( 'cProgress Demo\nwww.matpi.com' ) )
  end
  delete(h);
end
function busyExample
  disp ( 'creates a busy timer that will run for a given time' );
  f = figure;
  uip = uipanel ( 'parent', f, 'position', [0.3 0.3 0.4 0.4] );
  h = cProgress ('busy', 'Running Demo...', 'parent', uip );
  for ii=1:250
    cProgress ( 'busy', h );
    pause(0.0125);
    if ~ishandle ( h ); break; end
  end
  delete(f);
end
function busyTimerExample
  disp ( 'creates a busy progress bar with the bar updated every 0.25 seconds' );
  f = figure;
  uip = uipanel ( 'parent', f, 'position', [0.3 0.3 0.4 0.4] );
  cProgress ('busy', 'Running Demo...', 'parent', uip, 'timer', true, 'dTime', 0.25 );
  
end
